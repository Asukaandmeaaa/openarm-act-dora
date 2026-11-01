"""Pure MuJoCo load/physics tests, no renderer/device opened by these tests."""
import ast
from pathlib import Path
import mujoco
import numpy as np
import yaml
from deployment.common import ROOT
from deployment.config import names


def test_scene_is_self_contained_and_has_required_cameras_and_objects():
    model=mujoco.MjModel.from_xml_path(str(ROOT/'sim/scenes/act_fixed_20261015/scene.xml'))
    for camera in ('camera_ceiling','camera_wrist_right','preview_overview'):
        assert model.camera(camera).id>=0
    for body in ('orange_cube','black_frame','table'):
        assert model.body(body).id>=0
    for side in ('right','left'):
        for name in names(side): assert model.joint(name).id>=0
    data=mujoco.MjData(model);mujoco.mj_step(model,data,nstep=10)
    assert data.time>0 and np.isfinite(data.qpos).all()


def test_sim_graph_cannot_spawn_physical_nodes():
    graph=yaml.safe_load((ROOT/'dataflow-sim.yml').read_text())
    roles={n['args'].split()[0] for n in graph['nodes']}
    assert roles=={'simulation','control','observer','client','executor'}
    assert all(n['path']=='scripts/node' for n in graph['nodes'])
    text=(ROOT/'deployment/simulation.py').read_text()
    for forbidden in ('SingleArmDriver(', 'OpenArm(', 'VideoCapture(', 'enable_all(', 'socket('):
        assert forbidden not in text


def test_sim_does_not_overwrite_joint_positions_to_fake_tracking():
    tree=ast.parse((ROOT/'deployment/simulation.py').read_text())
    for fn in ast.walk(tree):
        if isinstance(fn,ast.FunctionDef) and fn.name in ('apply','main'):
            for statement in ast.walk(fn):
                if isinstance(statement,ast.Assign):
                    assert all('qpos' not in ast.unparse(t) for t in statement.targets)


def test_sim_and_real_share_policy_components():
    real={n['id']:n for n in yaml.safe_load((ROOT/'dataflow.yml').read_text())['nodes']}
    sim={n['id']:n for n in yaml.safe_load((ROOT/'dataflow-sim.yml').read_text())['nodes']}
    for name in ('policy-client','executor'):
        assert sim[name]==real[name]
