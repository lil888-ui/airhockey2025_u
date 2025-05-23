import myenv2            # ← myenv ではなく myenv2 を読み込む
import gymnasium as gym
import cv2
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
from tqdm import tqdm

def make_env():
    # myenv2 側で "AirHockey2-v0" として環境登録している前提
    return gym.make("AirHockey2-v0")

def render(images):
    for img in images:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', img_bgr)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def train(model_path=None, train=True):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    num_envs = 8
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    env = VecMonitor(env, "logs2/monitor2")

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/",
        name_prefix="sac_airhockey2",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # SACモデルの作成／ロード
    if model_path is None:
        model = SAC("MlpPolicy", env, verbose=True, device="cuda")
    else:
        model = SAC.load(model_path, env=env, verbose=True, device="cuda")

    if train:
        model.learn(total_timesteps=1000000, callback=checkpoint_callback)
        model.save("./models/sac_airhockey2_final")

    # 評価用に単一環境を生成
    eval_env = make_env()
    obs, _ = eval_env.reset()
    images = []
    observations = []

    for _ in tqdm(range(1000)):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        observations.append(obs)
        images.append(eval_env.render())
        if done:
            obs, _ = eval_env.reset()

    render(images)

if __name__ == "__main__":
    # 例: 新規学習するなら train_path=None, train=True
    # 保存済みモデルを読み込んで評価のみなら train_path="./models/sac_airhockey2_final", train=False
    train(model_path=None, train=True)
