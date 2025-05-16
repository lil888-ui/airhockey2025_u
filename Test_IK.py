import mujoco
import mujoco.viewer
import numpy as np
import math
import time
import random

def main():
    urdf_path = "/workspace/ros2_ws/src/airhockey2025/ka/assets/main.xml"
    model = mujoco.MjModel.from_xml_path(urdf_path)
    data = mujoco.MjData(model)

    # パラメータ
    Kp_task = 3.0
    Kd = 0.5
    x_range = (-0.5, 0.5)
    y_range = (-0.3, 0.3)

    # ee_siteのIDを取得
    ee_site_name = "ee_site"
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("🚀 Viewer起動中。ウィンドウを閉じるか Ctrl+C で終了してください。")

        mujoco.mj_resetData(model, data)

        while viewer.is_running():
            print("⏳ ターゲット位置を生成...")

            # ターゲット位置（XYランダム、Z=2固定）
            target_xy = np.random.uniform([x_range[0], y_range[0]], [x_range[1], y_range[1]])
            #target_pos = np.array([target_xy[0], target_xy[1], 5.0])
            target_pos=np.array([3,3,-100])
            print(f"🎯 ターゲット位置: {target_pos}")

            # 目標位置に向かって100ステップ制御
            for i in range(100):
                mujoco.mj_forward(model, data)

                # 現在のエンドエフェクタ位置を取得
                ee_pos = data.site_xpos[ee_site_id]
                pos_err = target_pos - ee_pos

                # サイト位置のJacobianを取得
                J_pos = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, J_pos, None, ee_site_id)

                # 関節速度コマンド（P制御） → ctrlに入れる
                qvel_des = J_pos.T @ (Kp_task * pos_err)
                qvel = data.qvel[:model.nu]
                data.ctrl[:] = qvel_des[:model.nu] - Kd * qvel

                mujoco.mj_step(model, data)
                viewer.sync()

            print("✅ ターゲット達成、次のランダム目標へ")

if __name__ == "__main__":
    main()
