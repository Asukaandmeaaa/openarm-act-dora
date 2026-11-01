"""Build only the frozen KDL math, never a ROS node or CAN interface."""
from pathlib import Path
import subprocess
import argparse

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',action='store_true',help='build frozen old implementation for offline comparison')
    args=parser.parse_args()
    dest=ROOT/'assets/gravity'
    extra=['-DCOLLECTION_REFERENCE',str(dest/'dynamics.cpp')] if args.reference else []
    libraries=['-lurdf'] if args.reference else ['-lurdfdom_model']
    output=ROOT/'validation/libcollection_gravity_reference.so' if args.reference else dest/'libcollection_gravity.so'
    output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC',
        '-I'+str(dest/'include'),'-I/usr/include/eigen3',
        '-I/opt/ros/jazzy/include/kdl_parser','-I/opt/ros/jazzy/include/urdf',
        '-I/usr/include/urdfdom',str(dest/'bridge.cpp'),*extra,
        '-L/opt/ros/jazzy/lib','-Wl,--disable-new-dtags,-rpath,/opt/ros/jazzy/lib',
        '-lkdl_parser','-lorocos-kdl',*libraries,
        '-o',str(output)],check=True)
if __name__=='__main__': main()
