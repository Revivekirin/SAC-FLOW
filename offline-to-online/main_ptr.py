import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import shimmy
import glob, tqdm, wandb, os, json, random, time, jax
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.flax_utils import save_agent
from utils.datasets import Dataset, ReplayBuffer, TrajectoryReplayBuffer

from evaluation import evaluate
from agents import agents
import numpy as np

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'square-mh-low_dim', 'Environment (dataset) name.')

# flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task3-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

# config_flags.DEFINE_config_file('agent', 'agents/acfql_gru.py', lock_config=False)
config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_string('entity', 'sophia435256-robros', 'wandb entity')
flags.DEFINE_string('mode', 'online', 'wandb mode')

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")

class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

def main(_):
    agent_name = FLAGS.agent['agent_name']
    task_name = FLAGS.env_name
    base_exp_name = get_exp_name(FLAGS.seed)
    exp_name = f"{agent_name}_{task_name}_{base_exp_name}"
    run = setup_wandb(project='qc', group=FLAGS.run_group, name=exp_name, entity=FLAGS.entity, mode=FLAGS.mode)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    
    # data loading
    if FLAGS.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0
    
    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    ## config for PTR
    USE_PTR = True  # PTR 사용 여부
    PTR_PRIORITY_METRIC = "uqm_reward"  # "uqm_reward", "avg_reward", "min_reward"
    PTR_BETA_WEIGHT = 0.5  # Weighted target 계수 (robomimic: 0.5 추천)
    
    # Dataset 처리
    train_dataset = process_train_dataset(train_dataset)
    
    if USE_PTR:
        
        # Offline dataset을 trajectory로 로드
        trajectory_buffer = TrajectoryReplayBuffer(
            buffer_size=FLAGS.buffer_size,
            observation_space=env.observation_space,
            action_space=env.action_space,
            alpha=0.6,  # priority exponent
            beta=0.4,   # IS exponent
            priority_metric=PTR_PRIORITY_METRIC,
        )
        
        # Offline dataset 로드
        trajectory_buffer.load_offline_dataset({
            'observations': np.array(train_dataset['observations']),
            'actions': np.array(train_dataset['actions']),
            'rewards': np.array(train_dataset['rewards']),
            'next_observations': np.array(train_dataset['next_observations']),
            'dones': np.array(train_dataset['terminals']),
        })
        
        print(f"Loaded {len(trajectory_buffer.trajectories)} trajectories into PTR buffer")

    # handle dataset
    def process_train_dataset(ds):
        """
        Process the train dataset to 
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )
        
        if is_robomimic_env(FLAGS.env_name):
            penalty_rewards = ds["rewards"] - 1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = penalty_rewards
            ds = Dataset.create(**ds_dict)
        
        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return ds
    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    
    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                    for prefix in prefixes},
        wandb_logger=wandb,
    )

    offline_init_time = time.time()
    # ===== Offline Training with PTR =====
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
        log_step += 1
        
        if USE_PTR:
            # PTR backward sampling
            batch_samples, traj_indices = trajectory_buffer.sample_batch_backward(
                batch_size=config['batch_size'],
                discount=discount,
            )
            
            # ReplayBufferSamples를 agent 형식으로 변환
            batch = {
                'observations': batch_samples.observations.numpy(),
                'actions': batch_samples.actions.numpy(),
                'rewards': batch_samples.rewards.numpy(),
                'next_observations': batch_samples.next_observations.numpy(),
                'masks': 1.0 - batch_samples.dones.numpy(),
                'terminals': batch_samples.dones.numpy(),
                'valid': np.ones_like(batch_samples.rewards.numpy()),
            }
            
            # Sequence 형태로 reshape (horizon_length=1로 처리)
            for key in batch:
                if key in ['observations', 'actions', 'next_observations']:
                    batch[key] = batch[key][:, None, ...]  # [B, 1, ...]
                else:
                    batch[key] = batch[key][:, None]  # [B, 1]
        else:
            # 기존 sampling
            batch = train_dataset.sample_sequence(
                config['batch_size'], 
                sequence_length=FLAGS.horizon_length, 
                discount=discount
            )
        
        # Agent update with weighted target
        if USE_PTR and hasattr(agent, 'critic_loss_with_weighted_target'):
            # Weighted target 사용
            # Agent 내부에서 beta_weight 사용하도록 config 전달
            agent.config.update({'beta_weight': PTR_BETA_WEIGHT})
        
        agent, offline_info = agent.update(batch)


        if i % FLAGS.log_interval == 0:
            logger.log(offline_info, "offline_agent", step=log_step)
        
        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

        # eval
        if i == FLAGS.offline_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            # during eval, the action chunk is executed fully
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=log_step)

    # transition from offline to online
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )
        
    ob, _ = env.reset()
    
    action_queue = []
    action_dim = example_batch["actions"].shape[-1]

    # Online RL
    update_info = {}

    from collections import defaultdict
    data = defaultdict(list)
    online_init_time = time.time()

    # ===== Online Training with PTR =====
    if USE_PTR:
        # Online phase: 기존 ReplayBuffer 대신 TrajectoryReplayBuffer 사용
        replay_buffer = trajectory_buffer
    else:
        replay_buffer = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
        )
    
    ob, _ = env.reset()
    action_queue = []
    
    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)
        
        # Action 실행
        if len(action_queue) == 0:
            action = agent.sample_actions(observations=ob, rng=key)
            action_chunk = np.array(action).reshape(-1, action_dim)
            for action in action_chunk:
                action_queue.append(action)
        action = action_queue.pop(0)
        
        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Reward 조정
        if is_robomimic_env(FLAGS.env_name):
            int_reward = int_reward - 1.0
        if FLAGS.sparse:
            int_reward = (int_reward != 0.0) * -1.0
        
        # PTR에 transition 추가
        if USE_PTR:
            replay_buffer.add_transition_to_trajectory(
                obs=ob.reshape(1, -1),
                next_obs=next_ob.reshape(1, -1),
                action=action.reshape(1, -1),
                reward=np.array([[int_reward]]),
                done=np.array([[done]]),
                infos=[info]
            )
        else:
            # 기존 방식
            transition = dict(
                observations=ob,
                actions=action,
                rewards=int_reward,
                terminals=float(done),
                masks=1.0 - terminated,
                next_observations=next_ob,
            )
            replay_buffer.add_transition(transition)
        
        # Episode 종료 처리
        if done:
            ob, _ = env.reset()
            action_queue = []
        else:
            ob = next_ob
        
        # Training
        if i >= FLAGS.start_training:
            if USE_PTR:
                # PTR backward sampling
                all_samples = []
                all_traj_indices = []
                for _ in range(FLAGS.utd_ratio):
                    samples, traj_idx = replay_buffer.sample_batch_backward(
                        batch_size=config['batch_size'],
                        discount=discount,
                    )
                    all_samples.append(samples)
                    all_traj_indices.append(traj_idx)
                
                # Batch 구성 (기존 형식 유지)
                batch = self._construct_batch_from_samples(all_samples)
            else:
                # 기존 방식
                batch = replay_buffer.sample_sequence(
                    config['batch_size'] * FLAGS.utd_ratio,
                    sequence_length=FLAGS.horizon_length,
                    discount=discount
                )
                batch = jax.tree.map(lambda x: x.reshape((
                    FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)
            
            agent, update_info["online_agent"] = agent.batch_update(batch)
            
        if i % FLAGS.log_interval == 0:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

        if i == FLAGS.online_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=log_step)

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

    end_time = time.time()

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    if FLAGS.save_all_online_states:
        c_data = {"steps": np.array(data["steps"]),
                 "qpos": np.stack(data["qpos"], axis=0), 
                 "qvel": np.stack(data["qvel"], axis=0), 
                 "obs": np.stack(data["obs"], axis=0), 
                 "offline_time": online_init_time - offline_init_time,
                 "online_time": end_time - online_init_time,
        }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)

if __name__ == '__main__':
    app.run(main)
