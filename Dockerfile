# ========================================
# ベース：ROS2公式イメージ（Foxy＋desktop版）
# ========================================
FROM osrf/ros:foxy-desktop

# 非対話モード
ENV DEBIAN_FRONTEND=noninteractive

# ========================================
# 追加パッケージ
# ========================================
RUN apt-get update && apt-get install -y \
    git-lfs \
    nano \
    libgl1-mesa-glx \
    libglfw3 \
    libosmesa6-dev \
    libglew-dev \
    ffmpeg \
    wget \
    unzip \
    python3-pip \
    ros-foxy-xacro \
    x11-apps\
    python3-tk\
    && apt-get clean

# git-lfs初期化
RUN git lfs install

# ========================================
# Pythonパッケージ
# ========================================
RUN pip3 install --upgrade pip
RUN pip3 install 'numpy>=1.20' mujoco matplotlib imageio mediapy

# ========================================
# MuJoCo GPUレンダリング用設定
# ========================================
ENV MUJOCO_GL=egl

# ========================================
# 作業ディレクトリ設定
# ========================================
WORKDIR /workspace

# ========================================
# SHELLをbashに切り替え
# ========================================
SHELL ["/bin/bash", "-c"]

# ========================================
# crane_x7_descriptionをクローン (git-lfs含む)
# ========================================
RUN git clone https://github.com/rt-net/crane_x7_description.git && \
    cd crane_x7_description && git lfs pull

# ========================================
# ROS2ワークスペース作成してcolcon build
# ========================================
RUN mkdir -p /workspace/ros2_ws/src && \
    mv /workspace/crane_x7_description /workspace/ros2_ws/src/ && \
    cd /workspace/ros2_ws && \
    source /opt/ros/foxy/setup.bash && \
    colcon build --symlink-install

# ========================================
# urdf_generatedディレクトリ作成 ＋ xacro展開 (use_gazebo:=false を指定)
# ========================================
RUN source /opt/ros/foxy/setup.bash && \
    source /workspace/ros2_ws/install/setup.bash && \
    mkdir -p /workspace/ros2_ws/src/crane_x7_description/urdf_generated && \
    xacro /workspace/ros2_ws/src/crane_x7_description/urdf/crane_x7.urdf.xacro use_gazebo:=false > /workspace/ros2_ws/src/crane_x7_description/urdf_generated/crane_x7.urdf

# ========================================
# ROS2環境自動source（便利用）
# ========================================
RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc && \
    echo "source /workspace/ros2_ws/install/setup.bash" >> ~/.bashrc

COPY GUI.py /workspace/ros2_ws/src/crane_x7_description/

