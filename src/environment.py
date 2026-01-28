from minigrid.minigrid_env import MiniGridEnv  
from minigrid.core.grid import Grid  
from minigrid.core.world_object import Goal, Wall 
from minigrid.core.mission import MissionSpace
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional

class Environment(MiniGridEnv):

    def __init__(self, grid_size=10, max_steps=100, n_seekers=1):
        self.n_seekers = n_seekers
        self.seeker_positions = []

        mission_space = MissionSpace(mission_func=lambda: "reach the goal")

        super().__init__(mission_space=mission_space, grid_size=grid_size, max_steps=max_steps)

        self.action_space = spaces.Discrete(8)

        self.action_map = {
            0: np.array([0, 1]),   # Right
            1: np.array([-1, 0]),  # Up
            2: np.array([0, -1]),  # Left
            3: np.array([1, 0]),   # Down
            4: np.array([1, 1]),   # Down-Right
            5: np.array([-1, 1]),  # Up-Right
            6: np.array([1, -1]),  # Down-Left
            7: np.array([-1, -1]), # Up-Left
        }

        self.observation_space = spaces.Dict({
            'agent': spaces.Box(0, grid_size-1, shape=(2,), dtype=np.int32),
            'goal': spaces.Box(0, grid_size-1, shape=(2,), dtype=np.int32),
            'seekers': spaces.Box(0, grid_size-1, shape=(n_seekers, 2), dtype=np.int32),
        })

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Optional outer walls
        # self.grid.wall_rect(0, 0, width, height)
        
        # TODO: Implement seekers and possibly static obstacles

        # Place goal in bottom-right corner
        self.goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_pos)
        
        # Place agent in top-left corner
        self.agent_pos = (1, 1)
        self.agent_dir = 0

    def gen_obs(self):
        # Return custom observation
        return {
            'agent': np.array(self.agent_pos, dtype=np.int32),
            'goal': np.array(self.goal_pos, dtype=np.int32),
            'seekers': np.array(self.seeker_positions, dtype=np.int32),
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed)
        return obs, info

    def step(self, action):
        # Get movement direction
        direction = self.action_map[action]

        # Calculate new position
        new_pos = (
            self.agent_pos[0] + direction[0],
            self.agent_pos[1] + direction[1]
        )

        # Clip to grid boundaries
        new_pos = (
            np.clip(new_pos[0], 0, self.width - 1),
            np.clip(new_pos[1], 0, self.height - 1)
        )

        # Check if cell is passable
        cell = self.grid.get(*new_pos)
        if cell is None or cell.can_overlap():
            self.agent_pos = new_pos
        
        # Increment step counter
        self.step_count += 1

        # Check if reached goal
        if tuple(self.agent_pos) == self.goal_pos:
            reward = 100.0
            terminated = True
        else:
            reward = -1.0
            terminated = False

        # Check if max steps reached
        truncated = self.step_count >= self.max_steps

        observation = self.gen_obs()
        info = {}
        
        return observation, reward, terminated, truncated, info