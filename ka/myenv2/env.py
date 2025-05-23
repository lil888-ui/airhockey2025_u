import mujoco
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from gymnasium.spaces import Box
from typing import Dict, Union

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}

class MyRobotEnv(MujocoEnv):
    def __init__(
        self,
        xml_path: str = "/workspace/ros2_ws/src/airhockey2025/ka/assets/main.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        **kwargs,
    ):
        # 1) モデル読み込みして観測次元取得
        model = mujoco.MjModel.from_xml_path(xml_path)
        obs_dim = model.nq + model.nv
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # 2) 行動空間：関節速度 [-1,1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=model.actuator_actnum.shape,
            dtype=np.float32,
        )

        # 3) 親クラス初期化
        super().__init__(
            model_path=xml_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode="rgb_array",
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # メタデータ（レンダー設定）
        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        # 各種パラメータ
        self.step_cnt_threshold = 200  # 最大ステップ数
        # 手先ターゲット用の初期化は reset_model() 内で行う

    def step(self, action):
        self.step_cnt += 1

        # action は正規化されているので実際の ctrlrange にスケール
        scaled_action = action * self.model.actuator_ctrlrange[:, 1]
        self.do_simulation(scaled_action, self.frame_skip)

        obs = self._get_obs()
        reward, done, info = self._compute_reward_and_done()
        truncated = self.step_cnt >= self.step_cnt_threshold

        return obs, reward, done, truncated, info

    def _get_obs(self):
        # qpos（位置）と qvel（速度）を連結
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def _compute_reward_and_done(self):
        # 手先座標（ee_site）
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        ee_pos = self.data.site_xpos[ee_id]  # np.array([x, y, z])

        # 目標座標（reset_model で設定済み）
        target = self.target_pos  # np.array([x, y, z])

        # 1) 距離報酬：目標との距離が小さいほど大きい
        dist = np.linalg.norm(ee_pos - target)
        r_dist = 1.0 - np.tanh(dist)

        # 2) 滑らかさペナルティ（速度ノルムを抑制）
        vel_norm = np.linalg.norm(self.data.qvel)
        r_smooth = -0.01 * vel_norm

        # 3) 時間ペナルティ
        r_time = -0.005

        # 4) 成功ボーナス
        if dist < 0.05:
            r_goal = +5.0
            done = True
        else:
            r_goal = 0.0
            done = False

        # 5) 落下ペナルティ：手先が低すぎたら強制終了
        ee_z = ee_pos[2]
        if ee_z < 0.2:
            done = True

        reward = r_dist + r_smooth + r_time + r_goal
        info = {"dist": dist, "vel": vel_norm, "goal": done}
        return reward, done, info

    def reset_model(self):
        # エピソード開始時リセット
        self.step_cnt = 0

        # 1) 関節の初期状態ランダム化
        qpos = self.init_qpos + np.random.uniform(-0.1, 0.1, size=self.model.nq)
        qvel = self.init_qvel + np.random.uniform(-0.1, 0.1, size=self.model.nv)
        self.set_state(qpos, qvel)

        # 2) 目標位置ランダム設定（テーブル範囲内、z は固定）
        #    x, y: ±80% の範囲に, z: 常に 0.1m 上
        #    （テーブルサイズは configure_puck とは独立想定）
        table_size = 0.5  # 例：テーブル横幅が 1m の場合
        x = np.random.uniform(-table_size * 0.8, table_size * 0.8)
        y = np.random.uniform(-table_size * 0.8, table_size * 0.8)
        z = 0.1
        self.target_pos = np.array([x, y, z], dtype=np.float64)

        return self._get_obs()
