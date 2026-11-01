"""Load the operator's hardware YAML; no live hardware inspection."""
import os
from pathlib import Path
import yaml
import xml.etree.ElementTree as ET
import numpy as np
from .common import ROOT


def names(side):
    return [f'openarm_{side}_joint{i}' for i in range(1,8)]+[f'openarm_{side}_finger_joint1']


def settings():
    path=Path(os.environ.get('OPENARM_HARDWARE_CONFIG', ROOT/'assets/hardware.local.yaml'))
    if not path.is_absolute(): path=ROOT/path
    if 'OPENARM_HARDWARE_CONFIG' not in os.environ and not path.exists(): path=ROOT/'assets/hardware.yaml'
    return yaml.safe_load(path.read_text())


def urdf_envelope(side):
    """Sealed URDF limits in the deployed raw-motor convention (negative opening)."""
    side=side.removesuffix('_arm')
    if side not in ('left','right'):raise ValueError('invalid arm side')
    tree=ET.parse(ROOT/'assets/collection_robot.urdf').getroot()
    position=[];velocity=[];effort=[]
    for name in names(side):
        limit=tree.find(f"joint[@name='{name}']/limit")
        lo,hi=float(limit.attrib['lower']),float(limit.attrib['upper'])
        if side=='left' and 'finger' in name:lo,hi=-hi,-lo
        position.append([lo,hi]);velocity.append(float(limit.attrib['velocity']))
        effort.append(float(limit.attrib['effort']))
    return np.array(position),np.array(velocity),np.array(effort)
