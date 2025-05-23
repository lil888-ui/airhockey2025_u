import myenv
import gymnasium as gym
import cv2
import os
from stable_baselines3 import A2C, SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
from tqdm import tqdm

def render(images):
    for img in images:
        # 画像をuint8のBGRに変換（OpenCV用）
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('Video', img_bgr)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def main():
    import gymnasium as gym
    from stable_baselines3 import A2C, SAC, PPO
    import numpy as np
    
    # モデルの保存ディレクトリを作成
    os.makedirs("models", exist_ok=True)
    
    env = gym.make("AirHockey-v0")
    
    # チェックポイントコールバックの設定
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # 10000ステップごとに保存
        save_path="./models/",
        name_prefix="ppo_airhockey",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    model = PPO("MlpPolicy", env, verbose=1, device="cuda")
    model.learn(total_timesteps=50000, callback=checkpoint_callback)
    
    # 最終モデルを保存
    model.save("./models/ppo_airhockey_final")
    
    vec_env = model.get_env()
    obs = vec_env.reset()
    images = []
    observations = []
    for i in tqdm(range(1000)):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        observations.append(obs)
        images.append(vec_env.render("rgb_array"))
    render(images)
    # import pickle
    # with open("observations.pkl", "wb") as f:
    #     pickle.dump(observations, f)

def make_env():
    import gymnasium as gym
    return gym.make("AirHockey-v0")

def train():
    import gymnasium as gym
    from stable_baselines3 import SAC, PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    import numpy as np
    from tqdm import tqdm
    
    # モデル保存用のディレクトリを作成
    os.makedirs("models", exist_ok=True)
    
    # 並列数を指定（CPUコア数に応じて調整）
    num_envs = 8
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    
    # チェックポイントコールバックの設定
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # 50000ステップごとに保存（環境数も考慮）
        save_path="./models/",
        name_prefix="sac_airhockey",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # モデルの作成と学習
    model = SAC("MlpPolicy", env, verbose=1, device="cuda")
    model.learn(total_timesteps=300_000, callback=checkpoint_callback)
    
    # 最終モデルを保存
    model.save("./models/sac_airhockey_final")
    
    # 評価用に単一環境を作成（並列環境では render できない）
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

def load_and_continue_training():
    """訓練済みモデルを読み込んで学習を続ける関数"""
    import gymnasium as gym
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    
    # モデル保存用のディレクトリを作成
    os.makedirs("models", exist_ok=True)
    
    # 環境の作成
    env = gym.make("AirHockey-v0")
    
    # 既存のモデルを読み込む（例：最後のチェックポイント）
    model_path = "./models/sac_airhockey_final"  # 適宜調整
    model = SAC.load(model_path, env=env)
    
    # チェックポイントコールバックの設定
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/continued/",
        name_prefix="sac_airhockey_continued",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # 続けて学習
    model.learn(total_timesteps=100_000, callback=checkpoint_callback)
    
    # 最終モデルを保存
    model.save("./models/continued/sac_airhockey_continued_final")

if __name__ == "__main__":
    # train()
    # トレーニング後にチェックポイントから再開したい場合は以下をコメント解除
    load_and_continue_training()