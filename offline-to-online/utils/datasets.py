from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self['next_observations']))

            batch['observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
        return batch

    def sample_sequence(self, batch_size, sequence_length, discount):
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)
        
        data = {k: v[idxs] for k, v in self.items()}

        # Pre-compute all required indices
        all_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]  # (batch_size, sequence_length)
        all_idxs = all_idxs.flatten()
        
        # Batch fetch data to avoid loops
        batch_observations = self['observations'][all_idxs].reshape(batch_size, sequence_length, *self['observations'].shape[1:])
        batch_next_observations = self['next_observations'][all_idxs].reshape(batch_size, sequence_length, *self['next_observations'].shape[1:])
        batch_actions = self['actions'][all_idxs].reshape(batch_size, sequence_length, *self['actions'].shape[1:])
        batch_rewards = self['rewards'][all_idxs].reshape(batch_size, sequence_length, *self['rewards'].shape[1:])
        batch_masks = self['masks'][all_idxs].reshape(batch_size, sequence_length, *self['masks'].shape[1:])
        batch_terminals = self['terminals'][all_idxs].reshape(batch_size, sequence_length, *self['terminals'].shape[1:])
        
        # Calculate next_actions
        next_action_idxs = np.minimum(all_idxs + 1, self.size - 1)
        batch_next_actions = self['actions'][next_action_idxs].reshape(batch_size, sequence_length, *self['actions'].shape[1:])
        
        # Use vectorized operations to calculate cumulative rewards and masks
        rewards = np.zeros((batch_size, sequence_length), dtype=float)
        masks = np.ones((batch_size, sequence_length), dtype=float)
        terminals = np.zeros((batch_size, sequence_length), dtype=float)
        valid = np.ones((batch_size, sequence_length), dtype=float)
        
        # Vectorized calculation
        rewards[:, 0] = batch_rewards[:, 0].squeeze()
        masks[:, 0] = batch_masks[:, 0].squeeze()
        terminals[:, 0] = batch_terminals[:, 0].squeeze()
        
        discount_powers = discount ** np.arange(sequence_length)
        for i in range(1, sequence_length):
            rewards[:, i] = rewards[:, i-1] + batch_rewards[:, i].squeeze() * discount_powers[i]
            masks[:, i] = np.minimum(masks[:, i-1], batch_masks[:, i].squeeze())
            terminals[:, i] = np.maximum(terminals[:, i-1], batch_terminals[:, i].squeeze())
            valid[:, i] = 1.0 - terminals[:, i-1]
        
        # Reorganize observations data format - maintain the exact same shape as the original function
        if len(batch_observations.shape) == 5:  # Visual data: (batch, seq, h, w, c)
            # Transpose to (batch, h, w, seq, c) format, consistent with the original function
            observations = batch_observations.transpose(0, 2, 3, 1, 4)  # (batch_size, h, w, sequence_length, c)
            next_observations = batch_next_observations.transpose(0, 2, 3, 1, 4)  # (batch_size, h, w, sequence_length, c)
        else:  # State data: maintain (batch, seq, state_dim) shape
            observations = batch_observations  # (batch_size, sequence_length, state_dim)
            next_observations = batch_next_observations  # (batch_size, sequence_length, state_dim)
        
        # Maintain the 3D shape of actions and next_actions, consistent with the original function
        actions = batch_actions  # (batch_size, sequence_length, action_dim)
        next_actions = batch_next_actions  # (batch_size, sequence_length, action_dim)
        
        return dict(
            observations=data['observations'].copy(),
            full_observations=observations,
            actions=actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
            next_actions=next_actions,
        )

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0

def add_history(dataset, history_length):

    size = dataset.size
    (terminal_locs,) = np.nonzero(dataset['terminals'] > 0)
    initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])
    assert terminal_locs[-1] == size - 1

    idxs = np.arange(size)
    initial_state_idxs = initial_locs[np.searchsorted(initial_locs, idxs, side='right') - 1]
    obs_rets = []
    acts_rets = []
    for i in reversed(range(1, history_length)):
        cur_idxs = np.maximum(idxs - i, initial_state_idxs)
        outside = (idxs - i < initial_state_idxs)[..., None]
        obs_rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs] * (~outside) + jnp.zeros_like(arr[cur_idxs]) * outside, 
            dataset['observations']))
        acts_rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs] * (~outside) + jnp.zeros_like(arr[cur_idxs]) * outside, 
            dataset['actions']))
    observation_history, action_history = jax.tree_util.tree_map(lambda *args: np.stack(args, axis=-2), *obs_rets),\
        jax.tree_util.tree_map(lambda *args: np.stack(args, axis=-2), *acts_rets)

    dataset = Dataset(dataset.copy(dict(
        observation_history=observation_history,
        action_history=action_history)))
    
    return dataset

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional
from flax.core.frozen_dict import FrozenDict


class TrajectoryReplayBuffer:
    """
    PTR을 위한 Trajectory 단위 저장 및 샘플링 버퍼 (JAX 버전)
    Dataset 클래스의 구조를 따르되, trajectory 단위 관리 추가
    """
    
    def __init__(
        self,
        buffer_size: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
        priority_metric: str = "uqm_reward",  # "uqm_reward", "avg_reward", "min_reward", "return"
        num_trajectories_to_sample: int = 256,
    ):
        """
        Args:
            buffer_size: 최대 trajectory 개수
            alpha: priority exponent
            beta: importance sampling exponent
            eps: numerical stability constant
            priority_metric: trajectory quality 계산 방식
            num_trajectories_to_sample: |B| in PTR paper
        """
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.priority_metric = priority_metric
        self.num_trajectories_to_sample = num_trajectories_to_sample
        
        # Trajectory 저장소
        self.trajectories: List[Dict[str, np.ndarray]] = []
        self.trajectory_priorities: List[float] = []
        
        # 현재 수집 중인 trajectory
        self.current_trajectory: Dict[str, List] = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'terminals': [],
            'masks': [],
        }
        
        # PTR sampling 상태 관리 (Figure 2)
        self.available_traj_indices: List[int] = []
        self.sampled_traj_indices: List[int] = []
        self.traj_sample_positions: Dict[int, int] = {}  # {traj_idx: current_backward_position}
        
    def add_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminal: bool,
        mask: float = 1.0,
    ) -> None:
        """현재 trajectory에 transition 추가"""
        self.current_trajectory['observations'].append(obs)
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['rewards'].append(reward)
        self.current_trajectory['next_observations'].append(next_obs)
        self.current_trajectory['terminals'].append(terminal)
        self.current_trajectory['masks'].append(mask)
        
        # Episode 종료 시 trajectory 저장
        if terminal:
            self._finalize_trajectory()
    
    def _finalize_trajectory(self) -> None:
        """현재 trajectory를 버퍼에 저장하고 priority 계산"""
        if len(self.current_trajectory['observations']) == 0:
            return
        
        # Numpy array로 변환
        traj = {
            k: np.array(v) for k, v in self.current_trajectory.items()
        }
        
        # Priority 계산
        priority = self._compute_trajectory_priority(traj)
        
        # 버퍼 저장
        self.trajectories.append(traj)
        self.trajectory_priorities.append(priority)
        self.available_traj_indices.append(len(self.trajectories) - 1)
        
        # 버퍼 크기 초과 시 오래된 trajectory 제거
        if len(self.trajectories) > self.buffer_size:
            self._remove_oldest_trajectory()
        
        # 현재 trajectory 초기화
        self.current_trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'terminals': [],
            'masks': [],
        }
    
    def _remove_oldest_trajectory(self) -> None:
        """가장 오래된 trajectory 제거 및 인덱스 업데이트"""
        removed_idx = 0
        self.trajectories.pop(removed_idx)
        self.trajectory_priorities.pop(removed_idx)
        
        # 모든 인덱스를 1씩 감소
        self.available_traj_indices = [
            idx - 1 for idx in self.available_traj_indices if idx > 0
        ]
        self.sampled_traj_indices = [
            idx - 1 for idx in self.sampled_traj_indices if idx > 0
        ]
        
        # traj_sample_positions 업데이트
        new_positions = {}
        for idx, pos in self.traj_sample_positions.items():
            if idx > 0:
                new_positions[idx - 1] = pos
        self.traj_sample_positions = new_positions
    
    def _compute_trajectory_priority(self, trajectory: Dict[str, np.ndarray]) -> float:
        """
        PTR Section 5.1: Trajectory quality 기반 priority 계산
        Robomimic sparse reward에 최적화
        """
        rewards = trajectory['rewards'].flatten()
        
        if self.priority_metric == "uqm_reward":
            # Upper Quartile Mean - 상위 25%의 평균
            if len(rewards) > 0:
                percentile_75 = np.percentile(rewards, 75)
                uqm_rewards = rewards[rewards >= percentile_75]
                return float(np.mean(uqm_rewards)) if len(uqm_rewards) > 0 else self.eps
            return self.eps
            
        elif self.priority_metric == "avg_reward":
            # 전체 평균
            return float(np.mean(rewards)) if len(rewards) > 0 else self.eps
            
        elif self.priority_metric == "min_reward":
            # 최소값 (conservative)
            return float(np.min(rewards)) if len(rewards) > 0 else self.eps
            
        elif self.priority_metric == "return":
            # Undiscounted return
            return float(np.sum(rewards))
            
        else:
            return 1.0
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        PTR의 핵심: Prioritized trajectory sampling + Backward sampling
        
        Returns:
            batch: 샘플링된 transitions의 딕셔너리
        """
        batch_transitions = []
        batch_traj_indices = []
        
        for _ in range(batch_size):
            # Step 1: 새로운 trajectory가 필요한 경우 prioritized sampling
            if len(self.sampled_traj_indices) < self.num_trajectories_to_sample:
                self._sample_new_trajectories()
            
            # Step 2: Sampled trajectories에서 backward sampling
            if len(self.sampled_traj_indices) == 0:
                # Fallback: random trajectory
                if len(self.trajectories) == 0:
                    raise ValueError("No trajectories available for sampling")
                traj_idx = np.random.choice(len(self.trajectories))
            else:
                traj_idx = self.sampled_traj_indices[0]
            
            # 현재 trajectory에서 backward position 가져오기
            if traj_idx not in self.traj_sample_positions:
                # 새로 샘플링된 trajectory: 끝에서 시작
                traj_len = len(self.trajectories[traj_idx]['observations'])
                self.traj_sample_positions[traj_idx] = traj_len - 1
            
            pos = self.traj_sample_positions[traj_idx]
            traj = self.trajectories[traj_idx]
            
            # Transition 추출
            transition = {
                'observations': traj['observations'][pos],
                'actions': traj['actions'][pos],
                'rewards': traj['rewards'][pos],
                'next_observations': traj['next_observations'][pos],
                'terminals': traj['terminals'][pos],
                'masks': traj['masks'][pos],
            }
            batch_transitions.append(transition)
            batch_traj_indices.append(traj_idx)
            
            # Backward position 업데이트
            self.traj_sample_positions[traj_idx] -= 1
            
            # Trajectory 끝에 도달하면 제거
            if self.traj_sample_positions[traj_idx] < 0:
                self.sampled_traj_indices.remove(traj_idx)
                del self.traj_sample_positions[traj_idx]
                # Available로 다시 추가
                if traj_idx not in self.available_traj_indices:
                    self.available_traj_indices.append(traj_idx)
        
        # 배치로 변환
        batch = self._transitions_to_batch(batch_transitions)
        batch['trajectory_indices'] = np.array(batch_traj_indices)
        
        return batch
    
    def _sample_new_trajectories(self) -> None:
        """
        PTR Section 5: Priority 기반 trajectory sampling
        """
        if len(self.available_traj_indices) == 0:
            return
        
        # Priority 계산
        available_priorities = np.array([
            self.trajectory_priorities[idx] for idx in self.available_traj_indices
        ])
        
        # Rank-based probability (PTR Eq. 5)
        ranks = np.argsort(np.argsort(-available_priorities)) + 1  # 높은 priority = 낮은 rank
        p_rank = 1.0 / ranks
        p_rank = p_rank ** self.alpha
        p_rank = p_rank / p_rank.sum()
        
        # Sample trajectories
        num_to_sample = min(
            self.num_trajectories_to_sample - len(self.sampled_traj_indices),
            len(self.available_traj_indices)
        )
        
        sampled_indices = np.random.choice(
            len(self.available_traj_indices),
            size=num_to_sample,
            replace=False,
            p=p_rank
        )
        
        for idx in sampled_indices:
            traj_idx = self.available_traj_indices[idx]
            self.sampled_traj_indices.append(traj_idx)
        
        # Available에서 제거
        for idx in sorted(sampled_indices, reverse=True):
            self.available_traj_indices.pop(idx)
    
    def _transitions_to_batch(self, transitions: List[Dict]) -> Dict[str, np.ndarray]:
        """Transition list를 batch 딕셔너리로 변환"""
        batch = {}
        keys = transitions[0].keys()
        
        for key in keys:
            batch[key] = np.stack([t[key] for t in transitions])
        
        return batch
    
    def load_from_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        """
        Offline dataset을 trajectory 단위로 로드
        
        Args:
            dataset: 'observations', 'actions', 'rewards', 'terminals', 'masks' 포함
        """
        # Episode boundaries 찾기
        terminals = dataset.get('terminals', None)
        if terminals is None:
            raise ValueError("Dataset must contain 'terminals'")
        
        episode_starts = [0]
        for i in range(len(terminals)):
            if terminals[i]:
                episode_starts.append(i + 1)
        
        # 마지막 episode가 완료되지 않았으면 제외
        if episode_starts[-1] >= len(terminals):
            episode_starts = episode_starts[:-1]
        
        # 각 episode를 trajectory로 저장
        for start_idx in range(len(episode_starts) - 1):
            start = episode_starts[start_idx]
            end = episode_starts[start_idx + 1]
            
            traj = {
                'observations': dataset['observations'][start:end],
                'actions': dataset['actions'][start:end],
                'rewards': dataset['rewards'][start:end].reshape(-1),
                'next_observations': dataset.get(
                    'next_observations', 
                    np.concatenate([dataset['observations'][start+1:end], 
                                   dataset['observations'][end-1:end]], axis=0)
                ),
                'terminals': terminals[start:end],
                'masks': dataset.get('masks', np.ones(end - start, dtype=np.float32)),
            }
            
            priority = self._compute_trajectory_priority(traj)
            self.trajectories.append(traj)
            self.trajectory_priorities.append(priority)
            self.available_traj_indices.append(len(self.trajectories) - 1)
    
    @property
    def size(self) -> int:
        """현재 저장된 trajectory 개수"""
        return len(self.trajectories)
    
    def get_statistics(self) -> Dict[str, float]:
        """버퍼 통계 정보"""
        if len(self.trajectories) == 0:
            return {
                'num_trajectories': 0,
                'avg_trajectory_length': 0.0,
                'avg_priority': 0.0,
                'max_priority': 0.0,
                'min_priority': 0.0,
            }
        
        traj_lengths = [len(t['observations']) for t in self.trajectories]
        
        return {
            'num_trajectories': len(self.trajectories),
            'avg_trajectory_length': np.mean(traj_lengths),
            'avg_priority': np.mean(self.trajectory_priorities),
            'max_priority': np.max(self.trajectory_priorities),
            'min_priority': np.min(self.trajectory_priorities),
            'num_available': len(self.available_traj_indices),
            'num_sampled': len(self.sampled_traj_indices),
        }