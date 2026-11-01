# Independent OpenArm ACT Dora deployment

- Runtime is self-contained here; never import runtime from another workspace.
- Read README.md, PORTING_NOTES.txt and docs/HARDWARE.md before hardware changes.
- Preserve collection Kp/Kd, right reference velocity and G(q_feedback) at 1.0,
  left position hold with gravity, and gripper position-only control.
- Preserve official executor and ACT math. Do not add gains, compensation,
  Ruckig, queues or different blending without explicit operator approval.
- Record every upstream patch in PORTING_NOTES.txt and freeze_sources.py output.
- Offline checks must never create CAN sockets, open cameras, construct a real
  robot or run dataflow.yml. Only the operator enables or starts hardware.
- ROS and Dora must never control one CAN interface simultaneously.
- Keep source/model/processor hashes and failed session logs; never reuse requests.
