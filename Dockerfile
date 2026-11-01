# Linux amd64 / Ubuntu 24.04 / Python 3.12. ROS supplies KDL libraries only.
FROM ros:jazzy-ros-base@sha256:8eed0a4669bbf43ee331878a4c39bd8ebddea8e3312a58b4fae8edb14526b617
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-venv python3-dev git build-essential pkg-config libeigen3-dev liborocos-kdl-dev liburdfdom-dev libcli11-dev ros-jazzy-kdl-parser libgl1 libegl1 libglfw3 libglib2.0-0 iproute2 kmod ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/openarm
COPY requirements/ requirements/
COPY scripts/install_python.sh scripts/install_python.sh
ARG TORCH_BACKEND=cpu
RUN ./scripts/install_python.sh "$TORCH_BACKEND"
COPY sources.lock.json sources.lock.json
COPY patches/ patches/
COPY scripts/ scripts/
COPY deployment/ deployment/
COPY assets/ assets/
RUN ./scripts/setup.sh "$TORCH_BACKEND" --skip-python
COPY . .
ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
ENV HF_HUB_OFFLINE=1
ENV OMP_NUM_THREADS=1
ENV MUJOCO_GL=egl
# No device mounts, host networking, privileged mode, or hardware entrypoint.
ENTRYPOINT []
CMD ["./scripts/validate.sh"]
