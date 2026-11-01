"""Same sensor modes as collection, latest frames; no ROS or motor imports."""
import json
import os
from pathlib import Path
import threading
import time
import cv2
import numpy as np
import dora
import pyarrow as pa
from .common import ROOT, Log, write_json, session


def main(role):
    cv2.setNumThreads(1)
    node=dora.Node(); log=Log('camera_'+role)
    lock=threading.Lock(); latest=[None]; running=threading.Event(); running.set()
    error=[]; capture=None; pipeline=None
    camera_config=Path(os.environ.get('OPENARM_CAMERA_CONFIG', ROOT/'assets/cameras.local.json'))
    if not camera_config.is_absolute(): camera_config=ROOT/camera_config
    if not camera_config.is_file(): raise ValueError('Copy assets/cameras.example.json to assets/cameras.local.json and configure your camera IDs')
    cameras=json.loads(camera_config.read_text())
    if role=='ceiling':
        import pyrealsense2 as rs
        pipeline=rs.pipeline(); config=rs.config(); config.enable_device(cameras['ceiling_serial'])
        config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
        config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
        pipeline.start(config)
        def read():
            frame=pipeline.wait_for_frames(1000).get_color_frame()
            if not frame: raise ValueError('missing RealSense color frame')
            return np.asanyarray(frame.get_data()).copy()
    else:
        path=cameras['wrist_right_path']
        capture=cv2.VideoCapture(path,cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH,640); capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        capture.set(cv2.CAP_PROP_FPS,60); capture.set(cv2.CAP_PROP_BUFFERSIZE,1)
        if (not capture.isOpened() or capture.get(cv2.CAP_PROP_FRAME_WIDTH)!=640
                or capture.get(cv2.CAP_PROP_FRAME_HEIGHT)!=480
                or abs(capture.get(cv2.CAP_PROP_FPS)-60)>.1
                or int(capture.get(cv2.CAP_PROP_FOURCC))!=cv2.VideoWriter_fourcc(*'MJPG')):
            raise ValueError('wrist camera capture negotiation failed')
        def read():
            ok,frame=capture.read()
            if not ok: raise ValueError('wrist capture failed')
            return frame
    def worker():
        try:
            sequence=0
            while running.is_set():
                frame=read(); stamp=time.monotonic_ns(); sequence+=1
                with lock: latest[0]=(frame,stamp,sequence)
        except Exception as exc: error.append(str(exc))
    thread=threading.Thread(target=worker,daemon=True); thread.start()
    try:
        for event in node:
            if event['type']!='INPUT': continue
            if error: raise ValueError(error[0])
            with lock: sample=latest[0]
            if sample is None: continue
            frame,stamp,sequence=sample
            if frame.shape!=(480,640,3):
                raise ValueError('camera frame shape invalid')
            ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,90])
            if not ok: raise ValueError('JPEG encoding failed')
            metadata=dict(source_monotonic_ns=stamp,source_sequence=sequence)
            node.send_output('image',pa.array(encoded.ravel()),metadata)
            log('frame',**metadata)
            write_json(session()/f'camera_{role}.json',dict(monotonic_ns=stamp,sequence=sequence))
    finally:
        running.clear(); thread.join(timeout=2)
        if pipeline is not None: pipeline.stop()
        if capture is not None: capture.release()
