import mujoco
import mujoco.viewer
import numpy as np
import math
import time
import random
import os
from pathlib import Path

def main():
    urdf_path = "/workspace/ros2_ws/src/airhockey2025/ka/assets/main.xml"
    model = mujoco.MjModel.from_xml_path(urdf_path)
    data = mujoco.MjData(model)

    # 発射設定
    x_range = (-0.8, 0.8)
    y_range = (-0.5, 0.5)
    speed = 1.0
    x_limit = 100  #1.1
    y_limit = 100  #0.6
    min_speed = 0.000000001  # 停止判定

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("🚀 Viewer起動中。ウィンドウを閉じるか Ctrl+C で終了してください。")
        Kp = 10.0
        Kd = 1.0
        mujoco.mj_resetData(model, data)
        while viewer.is_running():

            print("⏳ 5秒待機中...")
            for i in range(50):
                if i%10 == 0:
                    target_qpos = np.random.uniform(-np.pi, np.pi, model.nu)
                q = data.qpos[:model.nu]
                qd = data.qvel[:model.nu]
                data.ctrl[:] = Kp * (target_qpos - q) - Kd * qd

                mujoco.mj_step(model, data)
                viewer.sync()
                #time.sleep(0.1)  # 高速ステップ用スリープ

            # ランダム発射
            x0 = random.uniform(*x_range)
            y0 = random.uniform(*y_range)
            theta = random.uniform(0, 2 * math.pi)
            vx = speed * math.cos(theta)
            vy = speed * math.sin(theta)

            data.qpos[0:3] = [x0, y0, 0.0]
            data.qvel[0:3] = [vx, vy, 0.0]

            print(f"🚀 発射: 位置=({x0:.2f}, {y0:.2f})、速度=({vx:.2f}, {vy:.2f})")

            # 発射後の運動
            step_counter = 0
            while viewer.is_running():
                mujoco.mj_step(model, data)
                if step_counter % 10 == 0:
                    viewer.sync()
                    print(f"[step {step_counter}] qvel = {data.qvel[:3]}")
                step_counter += 1

                x, y = data.qpos[0], data.qpos[1]
                vx, vy = data.qvel[0], data.qvel[1]
                speed_now = math.hypot(vx, vy)

                if abs(x) > x_limit or abs(y) > y_limit:
                    print(f"🛑 場外: x={x:.2f}, y={y:.2f} → リセット")
                    break
                if speed_now < min_speed:
                    print(f"🛑 停止検知: 速度={speed_now:.4f} → リセット")
                    break

if __name__ == "__main__":
    main()
