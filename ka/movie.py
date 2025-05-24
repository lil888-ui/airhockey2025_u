import myenv2  # 環境登録済みモジュールをインポート
import gymnasium as gym
import cv2
import os
from stable_baselines3 import SAC
from tqdm import tqdm

# --- 設定 ---
MODEL_PATH = "./models/sac_airhockey2_final.zip"  # 既存モデルのパス
OUTPUT_VIDEO = "airhockey_10episodes.mp4"
FPS = 30
MAX_EPISODES = 10

# 動画保存関数
def save_video(frames, filename, fps=FPS):
    if not frames:
        print("No frames to save.")
        return
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    for img in frames:
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()
    print(f"Saved video to {filename}")

# 環境生成関数
def make_env():
    return gym.make("AirHockey2-v0")

if __name__ == '__main__':
    # モデルロード
    model = SAC.load(MODEL_PATH, device="cuda")

    # 評価環境
    eval_env = make_env()
    # レンダーFPSを明示的に設定して警告を回避
    eval_env.metadata['render_fps'] = FPS

    obs, _ = eval_env.reset()
    all_frames = []
    episode_count = 0

    # 指定エピソード数分をループ
    while episode_count < MAX_EPISODES:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        frame = eval_env.render()  # RGB ndarray
        all_frames.append(frame)

        if terminated or truncated:
            episode_count += 1
            obs, _ = eval_env.reset()
            print(f"Finished episode {episode_count}")

    # 動画としてまとめて保存
    save_video(all_frames, OUTPUT_VIDEO, fps=FPS)
