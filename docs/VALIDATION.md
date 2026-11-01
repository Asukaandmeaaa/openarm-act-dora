# Release software validation — 2026-11-01

A fresh Docker environment was built from the pinned Ubuntu 24.04 / ROS Jazzy base, with an isolated Python 3.12 venv. Upstream repositories were fetched at the published commits and the published patches applied. CAN Python bindings and the KDL gravity library were built from source; no original Conda environment or precompiled workspace library was reused.

The complete validation then ran with `--network none`, no devices or privileged mode, and a read-only mount containing the six pinned model files:

- Source, package, ACT algorithm-file, model and processor checks: passed.
- Native SDK/KDL binary and linked-library hashes checked: 16.
- `pip check`: no broken requirements.
- `pytest -q tests`: **197 passed in 4.66 s**.
- Real ACT inference on synthetic two-camera/state input: passed; CPU output **100×16**, all finite.
- Inference time in this smoke check: approximately 0.594 s. This is not a 30 Hz timing result or a task-performance benchmark.
- All 19 CUDA 12.8 dependency constraints resolved together with the common lock. A fresh CUDA installation was not executed in this release check.

The model was supplied read-only for this disconnected run. Public artifact download is a separate `fetch_model.py` step; archive and per-file hashes are checked before installation. The GitHub Actions workflow repeats installation and software validation on its Ubuntu runner, including the public model download.

During clean-install testing we found and fixed two portability problems: exported patches had lost trailing blank context lines, and the SDK source build needed `libcli11-dev`. Both are included in the published installer.

Nine critical local control/configuration/math files were also compared byte-for-byte with the original tested workspace: temporal execution/commit/client, feedforward, chunk settings/executor, hardware YAML, collection URDF, and gravity bridge. Existing upstream robot/driver collection-control patches are preserved and documented in PORTING_NOTES.txt. No physical graph, CAN socket, camera or motor was used by these checks.

Machine-readable summary: [validation.json](validation.json). Historical physical feedback is recorded separately in [EXPERIMENTS.md](EXPERIMENTS.md). Generated logs and machine-local binary seals stay outside Git.
