# Reproducing the deployment

## Supported scope / 复现范围

- Linux x86_64, Ubuntu 24.04, Python 3.12.
- Docker defaults to CPU for software tests. CUDA 12.8 Torch is available through `scripts/setup.sh cu128` or Docker `--build-arg TORCH_BACKEND=cu128`; the host NVIDIA driver/runtime must support that build.
- No pre-existing Conda environment, ROS workspace, camera, or robot is required for CPU validation.
- Unit tests use the published hardware configuration as a fixture; `actctl check` separately reads the operator configuration. Neither opens hardware.
- Tests verify ACT alignment/weights, 100/50 commitment and reset/STOP behavior, official executor AST parity, driver lifecycle through fakes, gravity math, scene loading/physics, and model/processor integrity.
- A synthetic inference proves loadability and output shape/finiteness. It does not evaluate grasping success or GPU timing on another computer.
- Training datasets and training scripts are outside this first deployment release; this is not a training-from-scratch reproduction.

## Preferred: Docker

Follow the three commands in the root README. Do not add `--privileged`, devices or host networking for validation. The container default command is `scripts/validate.sh`, not a hardware launcher.

Save local reports if desired:

```bash
mkdir -p validation
docker run --rm \
  -v openarm-act-models:/opt/openarm/models \
  -v "$PWD/validation:/opt/openarm/validation" openarm-act-dora
```

The image pins its base by digest, Python dependencies by version, upstream sources by commit, and model files by SHA256. Ubuntu apt security packages are resolved at build time; compiled-library hashes and installed package versions are sealed locally. This makes the source/model contract reproducible, not a promise of byte-identical binaries across compiler or system-library updates.

For an archived binary environment, save the successfully built image with `docker save`, retain its image ID/digest, and keep `runtime.local.json` with that image. Do not copy this file alone to a different native installation.

## Native setup

Install Ubuntu 24.04 and configure the official ROS Jazzy apt repository if it is absent. Only KDL/URDF development libraries are used here; no ROS controller is launched. Then:

```bash
./scripts/install_system_deps.sh
./scripts/setup.sh cpu  # or cu128 for NVIDIA
python3 scripts/fetch_model.py
./scripts/validate.sh
```

`setup.sh` creates an isolated `.venv`, installs locked Python dependencies, fetches exact upstream commits, applies the checked-in patches, builds `openarm-can` and the gravity library, installs the four Dora nodes, and seals the local environment. It does not probe cameras, create CAN sockets or enable motors.

Keep one backend per environment. To change CPU/CUDA backends, preserve existing `.venv` and `runtime.local.json` under different names and run setup into a fresh `.venv`. Never change dependencies during an active session.

The `cu128` selection pins Torch/Torchvision and the CUDA package versions in `requirements/cuda128.lock`, copied from the original tested environment. `runtime.local.json` records all installed Python package versions and local SDK/KDL binary hashes. The host NVIDIA driver remains machine-specific. CPU container validation does not claim a fresh GPU installation has been tested on another machine.

If a download is interrupted, rerun setup. If `bootstrap_sources.py` reports an unrecorded source edit, preserve/review it first; the installer intentionally does not reset a dirty checkout.

## Model download and offline transfer

`artifacts/model.json` is the authoritative artifact manifest. Six checkpoint/processor files are required; copying only `model.safetensors` is insufficient.

```bash
python3 scripts/fetch_model.py
python3 scripts/fetch_model.py --verify
# Alternatively download the release attachment elsewhere, transfer it, then:
python3 scripts/fetch_model.py --archive /path/to/act-sim-mix-30k.tar.gz
```

Existing files are verified rather than overwritten. A corrupted archive, unexpected archive member, symlink, or wrong file hash is rejected. All checkpoint weights include the ResNet backbone; initialization skips downloading a second backbone that would immediately be overwritten.

## Additional integration checks

On a machine fast enough for 30 Hz inference:

```bash
.venv/bin/python scripts/smoke_dataflow.py \
  --collection-feedforward --exercise-stop \
  --report validation/fake_graph.json
```

This script generates a separate graph with **all four physical camera/robot nodes replaced by fakes**. It preserves physical session pointers and requests. Default images are synthetic black frames; this is a transport/scheduling test, not a task score. CPU inference may miss the strict 33 ms slots and fail this real-time test even when the model smoke test passes.

`--saved-images` optionally reads `examples/images/train_episode0_ceiling.jpg` and `train_episode0_wrist_right.jpg`; supply your own compatible images. No recorded camera images are distributed. The exact filenames are also checked in `tests/fake_node.py`.

MuJoCo scene/physics loading is covered by tests. Optional full simulation:

```bash
./scripts/simctl test
# With a local graphical desktop:
./scripts/simctl view
```

Full simulation requires working EGL/GLFW and adequate inference/rendering speed. Scene geometry, contacts and camera positions are approximate, and do not establish physical performance.

## Source changes and provenance

`sources.lock.json` pins repository revisions. `scripts/freeze_sources.py` records raw binary diffs without trimming trailing context lines; it does not silently upgrade package versions or import machine-local paths. List every intentional upstream change in `PORTING_NOTES.txt`, regenerate patches, rebuild changed packages, and run `.venv/bin/python scripts/seal_environment.py`.

`./scripts/actctl prepare` freezes runtime code/config/model hashes for a new one-shot session. A source/config change after preparation requires a fresh session. Keep failed sessions and their one-shot requests for diagnosis; do not reuse them.

`docs/source_manifest.json` records the original tested source baseline, while Git history records the portability edits. It is provenance, not a claim that every release file is byte-identical to the original workspace.

## Network proxy (optional)

If your computer requires a proxy to reach GitHub/PyPI/PyTorch, configure it for the build process. Docker's predefined `HTTP_PROXY` and `HTTPS_PROXY` build arguments can forward existing proxy settings without recording them in the Dockerfile. A proxy bound only to host loopback needs an appropriate build-network configuration on Linux. Do not commit proxy credentials, `.env`, or local camera identifiers.

An easier offline test path is to download the model on the host with `scripts/fetch_model.py`, then mount it read-only:

```bash
docker run --rm --network none \
  -v "$PWD/models:/opt/openarm/models:ro" openarm-act-dora
```

Network availability is required for initial dependencies and source/model downloads, not for the subsequent software checks.
