"""Collection KDL gravity only; loading this module never touches hardware."""
import ctypes
import xml.etree.ElementTree as ET
import numpy as np
from .common import ROOT


class Gravity:
    def __init__(self, side, urdf=None):
        if side not in ('left','right'): raise ValueError('invalid arm side')
        urdf=ROOT/'assets/collection_robot.urdf' if urdf is None else urdf
        base=f'openarm_{side}_base_link'; tip=f'openarm_{side}_ee_base_link'
        joints={j.find('child').get('link'):j for j in ET.parse(urdf).getroot().findall('joint')}
        chain=[]; link=tip
        while link!=base:
            j=joints[link]
            if j.get('type')!='fixed': chain.append(j.get('name'))
            link=j.find('parent').get('link')
        if chain[::-1]!=[f'openarm_{side}_joint{i}' for i in range(1,8)]:
            raise ValueError('gravity chain joint order differs from collection')
        self.lib=ctypes.CDLL(str(ROOT/'assets/gravity/libcollection_gravity.so'))
        self.lib.gravity_create.argtypes=[ctypes.c_char_p]*3
        self.lib.gravity_create.restype=ctypes.c_void_p
        self.lib.gravity_destroy.argtypes=[ctypes.c_void_p]
        self.lib.gravity_destroy.restype=None
        self.lib.gravity_compute.argtypes=[ctypes.c_void_p]+[ctypes.POINTER(ctypes.c_double)]*2
        self.lib.gravity_compute.restype=ctypes.c_int
        self.ptr=self.lib.gravity_create(str(urdf).encode(),base.encode(),tip.encode())
        if not self.ptr: raise ValueError('collection KDL gravity initialization failed')

    def __call__(self, q):
        q=np.ascontiguousarray(q,dtype=np.float64)
        if q.shape!=(7,) or not np.isfinite(q).all(): raise ValueError('gravity needs 7 finite feedback angles')
        tau=np.empty(7,dtype=np.float64)
        ptr=ctypes.POINTER(ctypes.c_double)
        result=self.lib.gravity_compute(self.ptr,q.ctypes.data_as(ptr),tau.ctypes.data_as(ptr))
        if result: raise RuntimeError(f'KDL gravity failed: {result}')
        return tau

    def close(self):
        if getattr(self,'ptr',None): self.lib.gravity_destroy(self.ptr); self.ptr=None

    def __del__(self): self.close()
