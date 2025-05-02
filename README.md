# Air_Hockey_Challenge2025s

## rdp-ssh越しにコンテナからGUIを表示
    docker run -it --gpus=all --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all --env="DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix --rm mujoco_ros2_foxy_gpu:latest bash
