"""TEST GRAPH ONLY: official robot loop with a fake driver, synthetic cameras."""
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import pyarrow as pa
import dora

role,side=sys.argv[1:]
path=Path(os.environ['ACT_SESSION'])
if not path.name.startswith('software_smoke_'):
    raise ValueError('test node refuses production sessions')
node=dora.Node()
if role=='robot':
    from dora_openarm import main as official
    import openarm_can
    def forbid_can(*args,**kwargs): raise AssertionError('software graph must never create CAN')
    openarm_can.OpenArm=forbid_can
    class FakeDriver:
        def __init__(self,name):
            self.last_command=np.r_[np.zeros(7),-.02].astype(np.float32)
            self.last_command_dispatch_timestamp_ns=None
            self.moves=0
        def start(self): return True
        def stop(self): pass
        def send_position(self,q):
            self.last_command=np.asarray(q,dtype=np.float32)
            self.last_command_dispatch_timestamp_ns=time.time_ns()
            self.moves+=1
            (path/'fake_moves.txt').write_text(str(self.moves))
            return True
        def fetch_position(self,refresh=True): return self.last_command.copy()
        def fetch_state(self,refresh=True):
            return dict(qpos=self.last_command.copy(),qvel=np.zeros(8),qtorque=np.zeros(8),
                        tmos=np.full(8,25),trotor=np.full(8,25))
        def get_health(self): return ['ENABLED']*8, {}
    if os.environ.get('TEST_COLLECTION_FEEDFORWARD')=='1':
        from openarm_driver import SingleArmDriver, Config
        from deployment.feedforward import CollectionFeedforward
        from deployment.config import settings
        class CollectionFakeDriver(FakeDriver):
            def __init__(self,name):
                super().__init__(name)
                from types import SimpleNamespace
                self.config=object.__new__(Config);self.config._config=settings()
                self.arm_side=name;self.started=False;self._safety_stop_reason=None
                self._last_safety_warning_time_s=None;self.last_command_time_s=time.monotonic()
                self.joint_offsets=np.zeros(8);self.num_mit_motors=8;self.gripper_posforce=False
                self.kps=self.config.get_default_kps();self.kds=self.config.get_default_kds()
                self.latest_state=self.fetch_state()
                self.motion_commands=0
                self.openarm=SimpleNamespace(get_arm=lambda:SimpleNamespace(mit_control_all=self.capture))
                self.feedforward=CollectionFeedforward(name,self.config)
            def capture(self,params):
                self.moves+=1
                if side=='right' and any(p.dq!=0 for p in params):
                    self.motion_commands+=1
                    if self.motion_commands==50:(path/'fake_motion_exercised').touch()
                if side=='right': (path/'fake_moves.txt').write_text(str(self.moves))
            def start(self):self.started=True;return True
            def stop(self):
                self.started=False
                (path/f'fake_disabled_{side}.json').write_text(
                    __import__('json').dumps({'monotonic_ns':time.monotonic_ns(),'moves':self.moves}))
            def set_latest_state(self,**kwargs):self.latest_state=self.fetch_state()
            def send_position(self,q,**kwargs):return SingleArmDriver.send_position(self,q,**kwargs)
        FakeDriver=CollectionFakeDriver
    official.run(node,official.parse_args(['--side',side,'--no-align']),FakeDriver)
else:
    if os.environ.get('TEST_SAVED_IMAGES'):
        image_path=Path(os.environ['TEST_SAVED_IMAGES'])/f'train_episode0_{side}.jpg'
        pixels=cv2.imread(str(image_path))
        if pixels is None or pixels.shape!=(480,640,3):raise ValueError('invalid saved smoke-test image')
        image=np.frombuffer(image_path.read_bytes(),dtype=np.uint8)
    else:
        image=cv2.imencode('.jpg',np.zeros((480,640,3),dtype=np.uint8))[1]
    sequence=0
    for event in node:
        if event['type']!='INPUT': continue
        sequence+=1
        node.send_output('image',pa.array(image.ravel()),
                         {'source_monotonic_ns':time.monotonic_ns(),'source_sequence':sequence})
