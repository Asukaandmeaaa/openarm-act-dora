"""MuJoCo ONLY: copied fixed scene, real renderings and actuator physics.

No robot driver is constructed. Native scene physics/gains are preserved;
this is an approximate scene, not a calibrated digital twin of real motors.
"""
import json
import os
import time
from types import SimpleNamespace
import cv2
import dora
import mujoco
import numpy as np
import pyarrow as pa
from openarm_driver.driver import _create_default_checker, MAX_COMMAND_DT_S
from openarm_driver.config import Config
from .common import ROOT, session, Log, write_json, request
from .config import names, settings

SCENE=ROOT/'sim/scenes/act_fixed_20261015/scene.xml'


class World:
    def __init__(self):
        self.model=mujoco.MjModel.from_xml_path(str(SCENE))
        self.data=mujoco.MjData(self.model)
        self.indices={};self.controls={};self.dofs={}
        self.signs={'right':np.ones(8),'left':np.r_[np.ones(7),-1.]}
        for side in ('right','left'):
            joints=[self.model.joint(name).id for name in names(side)]
            self.indices[side]=self.model.jnt_qposadr[joints]
            self.dofs[side]=self.model.jnt_dofadr[joints]
            self.controls[side]=np.array([self.model.actuator(f'{side}_joint{i}_ctrl').id for i in range(1,8)]
                                       +[self.model.actuator(f'{side}_finger1_ctrl').id])
            q=np.r_[np.zeros(7),-.02]*self.signs[side]
            self.data.qpos[self.indices[side]]=q
            self.data.qpos[self.model.joint(f'openarm_{side}_finger_joint2').qposadr[0]]=q[7]
            self.data.ctrl[self.controls[side]]=q
        mujoco.mj_forward(self.model,self.data)
        self.renderers={name:mujoco.Renderer(self.model,height=480,width=640) for name in ('ceiling','wrist_right')}
        config=object.__new__(Config);config._config=settings()
        self.checker=_create_default_checker('right_arm',config)
        self.command=np.r_[np.zeros(7),-.02]
        self.last_command_at=time.monotonic()

    def state(self,side):
        return (self.data.qpos[self.indices[side]]*self.signs[side],
                self.data.qvel[self.dofs[side]]*self.signs[side])

    def apply(self,q,now):
        q=np.asarray(q,dtype=float)
        if q.shape!=(8,) or not np.isfinite(q).all():raise ValueError('invalid sim command')
        checked=self.checker.check(q,driver=SimpleNamespace(last_command=self.command),
                                   dt_s=min(max(now-self.last_command_at,0),MAX_COMMAND_DT_S))
        if not checked.is_safe:
            if checked.force_stop:raise ValueError(checked.message)
            if checked.fixed_joint_positions is not None:q=checked.fixed_joint_positions
        self.command=np.array(q,copy=True);self.last_command_at=now
        self.data.ctrl[self.controls['right']]=self.command
        return self.command.copy()

    def frames(self):
        out={}
        for role,r in self.renderers.items():
            r.update_scene(self.data,camera='camera_'+role)
            out[role]=r.render().copy()
        return out

    def close(self):
        for r in self.renderers.values():r.close()


def main():
    if not session().name.startswith('simulation_'):raise ValueError('sim node requires isolated simulation session')
    node=dora.Node();world=World();log=Log('simulation');viewer=None
    if os.environ.get('ACT_SIM_VIEWER')=='1':
        from mujoco import viewer as mjviewer
        viewer=mjviewer.launch_passive(world.model,world.data)
        viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid=world.model.camera('preview_overview').id
    for side in ('right','left'):node.send_output('status_'+side,pa.array(['started']))
    last_wall=time.monotonic();last_frame=0.;last_status=0.;sim_budget=0.;frames=0;moves=0
    try:
        for event in node:
            if event['type']!='INPUT':continue
            now=time.monotonic()
            if event['id']=='move_position':
                if (session()/'stop.request').exists():continue
                try:
                    raw=event['value'].field('qpos')[0].as_py()
                    final=world.apply(raw,now);moves+=1
                    log('sim_target',input_q=raw,final_q=final,metadata=event['metadata'])
                except (ValueError,KeyError) as exc:
                    request('stop',str(exc));log('sim_stopped',reason=str(exc))
                continue
            # Advance physics using wall-clock elapsed time, with explicit lag
            # reporting rather than teleporting qpos to each policy target.
            elapsed=now-last_wall;last_wall=now
            sim_budget+=min(elapsed,.1)
            steps=int(sim_budget/world.model.opt.timestep)
            if steps:
                mujoco.mj_step(world.model,world.data,nstep=steps)
                sim_budget-=steps*world.model.opt.timestep
            for side in ('right','left'):
                q,qd=world.state(side)
                node.send_output('position_'+side,pa.array([{'qpos':q.astype(np.float32)}],
                    type=pa.struct([('qpos',pa.list_(pa.float32()))])),
                    {'source_monotonic_ns':time.monotonic_ns()})
            if now-last_frame>=1/30:
                capture_ns=time.monotonic_ns()
                for role,rgb in world.frames().items():
                    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                    encoded=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,90])[1]
                    node.send_output('camera_'+role,pa.array(encoded.ravel()),{'source_monotonic_ns':capture_ns})
                    if frames==0:cv2.imwrite(str(session()/f'preview_{role}.png'),bgr)
                frames+=1;last_frame=now
            if viewer is not None:
                if not viewer.is_running():break
                viewer.sync()
            if now-last_status>.2:
                report=dict(simulation=True,hardware_accessed=False,physical_validation_passed=False,
                    sim_time_s=world.data.time,frames=frames,policy_targets=moves,
                    q_right=world.state('right')[0].tolist(),q_left=world.state('left')[0].tolist(),
                    cube_xyz=world.data.body('orange_cube').xpos.tolist(),
                    stopped=(session()/'stop.request').exists(),latest_wall_step_s=elapsed,
                    calibrated_dynamics=False,scene=str(SCENE))
                write_json(session()/'simulation_status.json',report);log('state',**report)
                last_status=now
    finally:
        if viewer is not None:viewer.close()
        world.close()
