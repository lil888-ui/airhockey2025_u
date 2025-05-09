# Air_Hockey_Challenge2025s

## ssh先のairhockey2025ディレクトリをマウント済みのコンテナ起動
    docker start affectionate_meitner

## rdp-ssh越しにコンテナからGUIを表示
    docker run -it --gpus=all --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all --env="DISPLAY=:21" -v /tmp/.X11-unix:/tmp/.X11-unix -v /storage/home/robot_dev2/airhockey2025:/workspace/ros2_ws/src/airhockey2025 affectionate_meitner bash

## rdp-ssh
    rdp-ssh -n my-gpu-desktop -a robot_dev2 start

# ホッケー台

    cd /workspace/ros2_ws/src
    git clone https://github.com/AirHockeyChallenge/air_hockey_challenge.git
    cd air_hockey_challenge
    git checkout tournament
    apt-get update
    apt-get install -y git-lfs
    git lfs install
    git lfs pull
    ls air_hockey_challenge/environments/data/iiwas/assets/
    export MUJOCO_GL=glfw

その後`hockey_test.py`を実行
