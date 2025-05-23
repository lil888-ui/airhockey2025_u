import mujoco
import mujoco.viewer
import numpy as np
import math
import time
import random

def damped_least_squares_torque_ik(model, data, ee_site_id, target_pos, viewer, Kp, Kd, lambda_, alpha=0.5):
    i = 0
    dt = model.opt.timestep
    jnt_range = model.jnt_range*0.95  # shape: (n_joints, 2)

    while np.linalg.norm(target_pos - data.site_xpos[ee_site_id]) > 0.0002:
        mujoco.mj_forward(model, data)

        # 現在のエンドエフェクタ位置を取得
        ee_pos = data.site_xpos[ee_site_id]
        error = target_pos - ee_pos

        if i == 0:
            print(f"⏳ 現在位置: {ee_pos}")
            print(f"ターゲットとの距離: {np.linalg.norm(error)}")

        lambda_fin = lambda_ * np.power(5, -np.linalg.norm(error))

        # サイト位置のJacobianを取得
        J_pos = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, J_pos, None, ee_site_id)

        # エンドエフェクタ速度
        ee_vel = J_pos @ data.qvel
        desired_acc = Kp * error - Kd * ee_vel

        # Damped Least Squares
        JJt = J_pos @ J_pos.T
        damped_inv = np.linalg.inv(JJt + lambda_fin**2 * np.eye(3))
        qvel_des = J_pos.T @ (damped_inv @ desired_acc)

        # ⚠️ ここで qvel_des を元に、次の位置 q + q̇ * dt を仮想計算
        q_next = data.qpos + qvel_des * dt

        # 関節の可動域にクリップ
        for j in range(model.nu):
            q_next[j] = np.clip(q_next[j], jnt_range[j, 0], jnt_range[j, 1])
        
        # クリップ後の q_next に対して、再び qvel_des を再計算（微分）
        qvel_des = (q_next - data.qpos) / dt

        # トルク制御に適用
        qvel = data.qvel[:model.nu]
        torque = qvel_des[:model.nu] - Kd * qvel
        data.ctrl[:] = torque

        mujoco.mj_step(model, data)
        viewer.sync()
        i = (i + 1) % 10000

    print("✅ ターゲット位置に到達")


def main():
    urdf_path = "/workspace/ros2_ws/src/airhockey2025/izumi/assets/crane_x7.xml"
    model = mujoco.MjModel.from_xml_path(urdf_path)
    data = mujoco.MjData(model)


    x_range = (-0.5, 0.5)
    y_range = (-0.3, 0.3)

    # ee_siteのIDを取得
    ee_site_name = "ee_site"
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("🚀 Viewer起動中。ウィンドウを閉じるか Ctrl+C で終了してください。")

        mujoco.mj_resetData(model, data)
        lambda_=0.001
        Kp=100.0
        Kd=1.0
        target_pos=np.array([-0.7,0.2,0.3])
        while viewer.is_running():
            print("⏳ ターゲット位置を生成...")

            # ターゲット位置（XYランダム、Z=2固定）
            #target_xy = np.random.uniform([x_range[0], y_range[0]], [x_range[1], y_range[1]])
            #target_pos = np.array([target_xy[0], target_xy[1], 5.0])
            mujoco.mj_forward(model, data)

            for alpha in np.linspace(0,1,num=10):
                print(data.site_xpos[ee_site_id])
                intermediate_pos = (1-alpha)*data.site_xpos[ee_site_id] + alpha*target_pos
                print(f"🎯 一時ターゲット位置: {intermediate_pos}")
                damped_least_squares_torque_ik(model, data, ee_site_id, intermediate_pos, viewer, Kp, Kd, lambda_, alpha)
            print("done")
            print(data.site_xpos[ee_site_id])
            break   

            #print("✅ ターゲット達成、次のランダム目標へ")

if __name__ == "__main__":
    main()
