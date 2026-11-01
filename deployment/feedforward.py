"""Optional collection control semantics. The official driver alone writes CAN.

Paired references dispatch on arrival. A 4ms timer refreshes gravity in hold;
reference velocity expires after its own planned sample interval.
"""
import json
import os
import time
from pathlib import Path
import numpy as np
from .gravity import Gravity
from .config import urdf_envelope
from .common import Log, request, write_json


def vector(value, name):
    value=np.asarray(value,dtype=np.float64)
    if value.shape!=(8,) or not np.isfinite(value).all():
        raise ValueError(name+' must contain 8 finite values')
    return value


class ReferenceVelocity:
    """Differentiate emitted, filtered references before any transport drops."""
    def __init__(self): self.previous={}; self.sequence=0

    def attach(self, name, q, metadata):
        metadata=dict(metadata or {})
        q=vector(q,'executor position')
        dt=int(metadata['reference_dt_ns'])/1e9
        if dt<=0: raise ValueError('reference interval must be positive')
        reset=metadata.get('reference_reset',False)
        previous=self.previous.get(name)
        velocity=np.zeros(8) if reset or previous is None else (q-previous)/dt
        if metadata.get('reference_end',False): velocity[:]=0
        velocity[7]=0
        self.previous[name]=q.copy(); self.sequence+=1
        metadata['reference_velocity_json']=json.dumps(velocity.tolist(),allow_nan=False)
        metadata['reference_sequence']=self.sequence
        return metadata


class CollectionFeedforward:
    def __init__(self, side, config, *, gravity=None, log=None, path=None):
        self.side=side.removesuffix('_arm'); self.config=config
        options=config._config['collection_feedforward']
        if options.get('gravity_scale')!=1.0:
            raise ValueError('collection mode requires the frozen gravity scale 1.0')
        if config.get_gripper_posforce() or np.any(config.get_joint_offsets(side)):
            raise ValueError('collection mode requires MIT gripper and frozen zero offsets')
        if config.get_motor_types()!=['DM8009']*2+['DM4340']*2+['DM4310']*4:
            raise ValueError('collection mode motor types differ from frozen hardware')
        self.gravity=Gravity(self.side) if gravity is None else gravity
        self.path=Path(os.environ['ACT_SESSION']) if path is None else Path(path)
        self.log=Log('feedforward_'+self.side) if log is None else log
        self.pending=None; self.held=False; self.reason=None; self.sequence=-1
        self.reference_expires=0.0
        self.metadata={}; self.last_tau=np.zeros(8); self.requested_position=None
        self.requested_velocity=None; self.position_limited_axes=[]
        self.velocity_limited_axes=[]; self.torque_limited_axes=[]; self.requested_torque=None
        self.limits,self.velocity_limits,self.effort_limits=urdf_envelope(side)
        if not np.array_equal(self.limits,config.get_joint_limits(side)):
            raise ValueError('hardware positions differ from sealed URDF motor-coordinate ranges')
        self.limits=np.clip(self.limits,-12.5,12.5)
        self.protocol_velocity=np.array([45,45,10,10,30,30,30,30.])
        self.protocol_torque=np.array([54,54,28,28,10,10,10,10.])
        for gains,maximum in [(config.get_default_kps(),500),(config.get_default_kds(),5)]:
            if gains.shape!=(8,) or not np.isfinite(gains).all() or np.any((gains<0)|(gains>maximum)):
                raise ValueError('MIT gain outside protocol range')

    def stopping(self):
        return (self.path/'stop.request').exists() or (self.path/'disable.request').exists()

    def hold(self):
        self.held=True; self.pending=None; self.metadata={}; self.reference_expires=0.0

    def fault(self, reason):
        self.hold()
        if self.reason is None:
            self.reason=str(reason)
            request('stop',self.reason)
            fault_path=self.path/f'{self.side}_fault.json'
            if not fault_path.exists(): write_json(fault_path,{'reason':self.reason})
            self.log('reference_rejected',reason=self.reason)

    def receive(self, position, metadata):
        if self.held or self.stopping(): self.hold(); return
        try:
            seq=metadata['reference_sequence']
            if type(seq) is not int: raise ValueError('reference sequence must be an integer')
            if seq<=self.sequence: return
            interval=metadata['reference_dt_ns']
            if type(interval) is not int or interval!=4_000_000:
                raise ValueError('reference interval differs from configured 250Hz executor')
            q=vector(position,'reference position')
            v=vector(json.loads(metadata['reference_velocity_json']),'reference velocity')
            if self.side!='right': raise ValueError('left policy motion is not enabled')
            if v[7]!=0: raise ValueError('gripper reference velocity must be zero')
            self.sequence=seq; self.pending=(q.copy(),v.copy(),dict(metadata))
        except (KeyError,ValueError,TypeError) as exc: self.fault(exc)

    def tick(self, driver):
        if not getattr(driver,'_command_enabled',driver.started) or (self.path/'disable.request').exists(): return
        if self.stopping(): self.hold()
        if self.pending is None and not self.held and time.monotonic()<self.reference_expires:
            return  # Do not erase an unexpired velocity due to timer phase.
        driver.set_latest_state(timeout_us=300)
        if self.pending is not None and not self.held:
            q,v,self.metadata=self.pending; self.pending=None
            self.reference_expires=time.monotonic()+self.metadata['reference_dt_ns']/1e9
        else:
            q=driver.last_command.copy(); v=np.zeros(8)
        driver.send_position(q,velocity=v)

    def parameters(self, driver, position, velocity):
        """Clamp physical endpoints, pair dq, and add gravity without a second PD."""
        if self.stopping(): self.hold()
        q=driver.last_command.copy() if self.held else position
        v=np.zeros(8) if self.held or velocity is None else velocity
        try:
            q=vector(q,'target position').copy(); v=vector(v,'target velocity').copy()
            self.requested_position=q.copy()
            self.requested_velocity=v.copy(); self.position_limited_axes=[]
            self.velocity_limited_axes=[]
            if v[7]!=0: raise ValueError('gripper velocity feedforward is not supported')
            vmax=np.minimum(self.velocity_limits,self.protocol_velocity)
            bounded_v=np.clip(v,-vmax,vmax)
            self.velocity_limited_axes=(np.flatnonzero(bounded_v!=v)+1).tolist()
            v=bounded_v
            bounded=np.clip(q,self.limits[:,0],self.limits[:,1])
            changed=bounded!=q
            self.position_limited_axes=(np.flatnonzero(changed)+1).tolist()
            q=bounded
            # A saturated position is stationary on that axis, so its unbounded
            # trajectory derivative must not continue pushing past the endpoint.
            v[changed]=0.
            at_boundary=(q<=self.limits[:,0])&(v<0) | (q>=self.limits[:,1])&(v>0)
            v[at_boundary]=0.
        except (ValueError,RuntimeError,TypeError) as exc:
            self.fault(exc)
            q=vector(driver.last_command,'last dispatched position').copy(); v=np.zeros(8)
            if np.any(np.abs(q)>12.5): raise ValueError('no valid MIT hold position')
        try:
            tau=np.r_[self.gravity(vector(driver.latest_state['qpos'],'feedback')[:7]),0.]
            if not np.isfinite(tau).all():raise ValueError('nonfinite gravity torque')
            self.requested_torque=tau.copy()
            tmax=np.minimum(self.effort_limits,self.protocol_torque)
            bounded_tau=np.clip(tau,-tmax,tmax)
            self.torque_limited_axes=(np.flatnonzero(bounded_tau!=tau)+1).tolist()
            tau=bounded_tau
            self.last_tau=tau.copy()
        except (ValueError,RuntimeError,TypeError,KeyError) as exc:
            self.fault(exc)
            # Clear velocity at the last dispatched target. If gravity failed,
            # retain the last valid torque, explicitly logged as a fault hold.
            fallback=vector(driver.last_command,'last dispatched position')
            if np.any(np.abs(fallback)>12.5): raise ValueError('no valid MIT hold position')
            q=fallback.copy(); v=np.zeros(8); tau=self.last_tau.copy()
        return q,v,tau

    def dispatched(self, driver, q, v, tau):
        self.log('mit_command',q_ref=q,dq_ref=v,tau_ff=tau,
                 q_feedback=driver.latest_state['qpos'],
                 qd_feedback=driver.latest_state['qvel'],
                 dispatch_timestamp_ns=driver.last_command_dispatch_timestamp_ns,
                 requested_position=self.requested_position,
                 requested_velocity=self.requested_velocity,
                 position_limited_axes=self.position_limited_axes,
                 velocity_limited_axes=self.velocity_limited_axes,
                 requested_torque=self.requested_torque,
                 torque_limited_axes=self.torque_limited_axes,
                 reference_expires_monotonic_ns=int(self.reference_expires*1e9),
                 held=self.held,fault=self.reason,reference=self.metadata)
