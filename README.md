# Air_Hockey_Challenge2025s

## rdp-ssh越しにコンテナからGUIを表示
    docker run -it --gpus=all --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all --env="DISPLAY=:21" -v /tmp/.X11-unix:/tmp/.X11-unix -v /storage/home/robot_dev2/airhockey2025:/workspace/ros2_ws/src/airhockey2025 affectionate_meitner bash

## rdp-ssh
    rdp-ssh -n my-gpu-desktop -a robot_dev2 start
