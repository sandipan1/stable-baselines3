
import gym
print(__name__)
import numpy as np
import sys
import os
# from stable_baselines3 import  SAC, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
# from stable_baselines3.her import   HerReplayBuffer
from stable_baselines3.her import goal_selection_strategy
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
# from fetch_image_env import  make_sb3_env
import gym
from stable_baselines3.common.env_checker import check_env
from gym.wrappers import Monitor
from experiment_src.video import VideoRecorder
from stable_baselines3.custom_td3 import TD3 , CustomTD3Policy , CustomCombinedExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.custom_her.her_replay_buffer import HerReplayBuffer
from MakeTreeDir import MAKETREEDIR
from stable_baselines3.common.custom_callbacks import ProgressBarManager, EvalCallback
from stable_baselines3.common.monitor import Monitor
from experiment_src.custom_point_env import make_sb3_point_env



def train(env,work_dir):    
    model_dir = os.path.join(work_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    env= Monitor(env, os.path.join(model_dir, "monitor.csv"))

    feature_extractor_class = CustomCombinedExtractor
    feature_extractor_kwargs = dict(features_dim=128)

    # custom_td3_policy= CustomTD3Policy(env.observation_space, env.action_space,
                                    #    )
    policy_kwargs = {
        
        "features_extractor_class" : feature_extractor_class,
        "features_extractor_kwargs" : feature_extractor_kwargs,
        "normalize_images": False,
        "net_arch":[1024,1024]

    }
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(policy="CustomTD3Policy", env=env,learning_rate=1e-3,buffer_size=100000,
                replay_buffer_class=HerReplayBuffer,
        # Parameters for HER
                replay_buffer_kwargs=dict(
                    n_sampled_goal=4,
                    goal_selection_strategy=goal_selection_strategy,
                    online_sampling=online_sampling,
                    max_episode_length=max_episode_length,
                ),
                policy_kwargs=policy_kwargs,
                seed = 10,
                verbose=1,
                tensorboard_log=os.path.join(work_dir, "tensorboard_log"),
                embedding_space_distance= False,
                monitor_wrapper =True, 
                action_noise = action_noise
                )

    print(model)
    eval_callback = EvalCallback(eval_env=env,n_eval_episodes=5,eval_freq=5000, model_dir=model_dir)
    total_timesteps = 100001
    with ProgressBarManager(total_timesteps) as progress_callback: # this the garanties th,at the tqdm progress bar closes correctly
        # model.learn(2000, callback=callback), 
        model.learn(total_timesteps=total_timesteps, log_interval=1000, callback=[eval_callback,progress_callback])
    # model.save(os.path.join(work_dir, "model/td3"))

    print("model trained and saved")



def eval_and_save_video(env,work_dir):
    model = TD3.load(os.path.join(work_dir, "model/best_model"), env)

    count = 0
    rewards=[]
    episode_reward = 0
    obs = env.reset()
    v = VideoRecorder(video_dir="./video")
    v.init(enabled=True)
    for i in range(200):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        # env.render(mode="human")
        v.record(env)
        count+=1
        if done:
            print(count)
            rewards.append(episode_reward)
            episode_reward=0
            count=0
            obs = env.reset()

    print("Average reward on evaluation" , np.mean(np.array(rewards)))

    v.save(os.path.join(work_dir,"video/200.mp4"))


if __name__ == '__main__':
    goal_selection_strategy = "future"
    online_sampling = True
    max_episode_length = 100
    # env = make_sb3_env(env_name="fetch_reach", action_repeat=2, max_episode_steps=50, seed=10, fixed=False, reward_type="dense")
    # eval(env, model_path="td3_fetch")
    # eval(env=env, model_path="td3_fetch")
    env = make_sb3_point_env(seed=0)

    global_dir = os.path.abspath(os.getcwd() + "/../")
    work_dir = os.path.join(global_dir,"data/point_mass3")

    directory = MAKETREEDIR()
    directory.makedir(work_dir)
    print(os.getenv('SLURM_JOB_ID'))
    train(env, work_dir=work_dir)
