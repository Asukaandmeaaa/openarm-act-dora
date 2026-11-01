import ast
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time
from unittest.mock import Mock
import numpy as np
import pyarrow as pa
import pytest
import yaml
from deployment.common import ROOT, UPSTREAM
from deployment.config import settings
from deployment.cli import verify_model
from openarm_driver import SingleArmDriver


def test_model_and_hardware_contract():
    verify_model()
    cfg=settings()
    assert cfg['can_interface']=={'right_arm':'can0','left_arm':'can1'}
    assert cfg['joint_offsets']['right_arm']==[0.]*8
    assert cfg['control_gains']['kps']==[70.,70.,70.,60.,10.,10.,10.,5.]
    assert cfg['joint_limits']['right_arm'][7]==[-.7854,0.]
    assert cfg['control_gains']['kds']==[2.75,2.5,2.,2.,.7,.6,.5,.1]
    assert json.loads((ROOT/'assets/deployment.json').read_text())['control_mode']=='collection_feedforward'
    assert cfg['collection_feedforward']=={'enabled':True,'gravity_scale':1.0}


def test_graph_uses_official_components_no_ros_and_no_left_policy_motion():
    graph=yaml.safe_load((ROOT/'dataflow.yml').read_text())
    nodes={n['id']:n for n in graph['nodes']}
    assert 'move_position' not in nodes['robot-left']['inputs']
    assert nodes['robot-right']['inputs']['move_position']['source']=='control/move_position_right'
    assert nodes['control']['inputs']['move_position']['source']=='executor/move_position_right'
    assert 'ros' not in (ROOT/'dataflow.yml').read_text().lower()
    assert nodes['policy-client']['inputs']['observation']['queue_size']==1


def test_observer_only_training_cameras():
    from dora_openarm_observer.main import _reset_observation
    obs={}; _reset_observation(obs,['right','left'])
    assert set(obs)=={'arm_right','arm_left','camera_ceiling','camera_wrist_right','id'}


def test_official_executor_hermite_filter_and_reset_are_preserved():
    # The sole upstream executor change is propagation of metadata. Compare
    # its algorithm AST against the pinned git version after normalizing that.
    import subprocess
    path=UPSTREAM/'nodes/dora-openarm-actions-executor'
    relative='src/dora_openarm_actions_executor/main.py'
    original=subprocess.check_output(['git','-C',str(path),'show','HEAD:'+relative],text=True)
    current=(path/relative).read_text()
    class MetadataOnly(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if (isinstance(node.func,ast.Attribute) and node.func.attr=='send_output'
                and isinstance(node.args[0],ast.Constant)
                and node.args[0].value in ('move_position_right','move_position_left')):
                node.args[2]=ast.parse('{"timestamp": timestamp}',mode='eval').body
            return node
    assert ast.dump(MetadataOnly().visit(ast.parse(current)))==ast.dump(MetadataOnly().visit(ast.parse(original)))


def test_official_executor_emits_full_chunk_and_preserves_source_metadata():
    from dora_openarm_actions_executor.main import _main_executor
    class Done(Exception):pass
    captured=[]
    class Node:
        def send_output(self,name,value,metadata):
            captured.append((name,value,metadata))
            if len(captured)==10:raise Done()
    async def run():
        queue=asyncio.Queue()
        await queue.put(dict(value=pa.array(np.zeros((50,16)).tolist(),type=pa.list_(pa.float32())),
                             metadata=dict(interval=33_333_333,reset=True,cutoff_hz=15,
                                           observation_monotonic_ns=123,chunk_id='test')))
        with pytest.raises(Done):
            await _main_executor(Node(),queue,['right','left'],True,True,250.)
    asyncio.run(run())
    assert all(m['observation_monotonic_ns']==123 for _,_,m in captured)
    assert all(len(v.field('qpos')[0])==8 for _,v,_ in captured)
    right=[m for name,_,m in captured if name=='move_position_right']
