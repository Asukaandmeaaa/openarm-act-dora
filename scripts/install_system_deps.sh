#!/usr/bin/env bash
set -euo pipefail
# Ubuntu 24.04 with the official ROS Jazzy apt repository already configured.
# This only installs development libraries; it does not start ROS or configure CAN.
if [ "$(id -u)" = 0 ]; then apt_cmd=(); else apt_cmd=(sudo); fi
"${apt_cmd[@]}" apt-get update
"${apt_cmd[@]}" apt-get install -y python3-venv python3-dev git build-essential pkg-config \
  libeigen3-dev liborocos-kdl-dev liburdfdom-dev libcli11-dev ros-jazzy-kdl-parser \
  libgl1 libegl1 libglfw3 libglib2.0-0 iproute2 kmod ca-certificates
