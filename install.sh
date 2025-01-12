#!/bin/bash
apt-get update && apt-get install -y libgl1-mesa-dev
apt-get update && apt-get install -y libxxf86vm1
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

pip install opencv-python
pip install -r requirements.txt
pip uninstall -y basicsr
pip install git+https://github.com/piperwolters/BasicSR.git
