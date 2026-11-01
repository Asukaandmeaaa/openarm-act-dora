# Third-party sources

The root MIT license applies to original project contributions. Upstream-derived files and patches retain their source licenses. All fetched upstream repositories keep their original LICENSE files; `third_party/Apache-2.0.txt` provides the license for the vendored Apache-2.0 material.

| Component | Source and tracking | License |
| --- | --- | --- |
| Dora evaluation and four nodes | Enactic URLs and exact commits in `sources.lock.json`; changes in `patches/` | Apache-2.0 |
| OpenArm driver and CAN binding | Enactic URLs and commits in `sources.lock.json`; collection-control driver hooks and CAN `refresh_one` binding patch | Apache-2.0 |
| ACT policy and temporal ensemble | `lerobot==0.6.1`; ACT source hash in `sources.lock.json` | Apache-2.0 |
| Robot MuJoCo XML and meshes | `assets/openarm_mujoco/`, copied from the original fixed scene bundle; original LICENSE included | Apache-2.0 |
| Gravity reference implementation | Local `openarm_hardware` recovery snapshot; original file hashes in `assets/gravity/provenance.json` | Apache-2.0, inherited from source package |
| Robot description and limits | OpenArm description used in the expert-collection configuration; frozen `assets/collection_robot.urdf` and `moveit_joint_limits.yaml` | Apache-2.0, OpenArm upstream-derived |
| Official sample dataflow fixture | Pinned Dora evaluation source, `tests/reference/dataflow-cell-sample.yaml` | Apache-2.0 |
| Earlier executor regression fixture | Local deployment snapshot, `tests/reference/chunk_executor_full50.py`; uses upstream math by import | Original integration under MIT |

The vendored MuJoCo XML and collection URDF reflect local scene/control integration. They should not be assumed identical to the latest upstream assets. Git records the exact published files; do not silently substitute assets from a current upstream branch.

Python and native dependencies (including PyTorch, NumPy, MuJoCo, KDL and ROS libraries) remain separate dependencies under their distributed licenses. See package metadata and the corresponding upstream license for redistribution of a built image.
