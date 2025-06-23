import ai2thor
from ai2thor.controller import Controller
import numpy as np
import torch
import torch.nn as nn
import cv2
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import random
import gym
from gym import spaces

class AI2ThorNoMaDEnv(gym.Env):
    """
    AI2Thor environment wrapper for NoMaD-RL training
    Supports both goal-conditioned navigation and exploration modes
    """
    
    def __init__(
        self,
        scene_names: List[str] = None,
        image_size: Tuple[int, int] = (96, 96),
        max_episode_steps: int = 500,
        success_distance: float = 1.0,
        rotation_step: int = 90,
        movement_step: float = 0.25,
        context_size: int = 5,
        goal_prob: float = 0.5,  # Probability of goal-conditioned vs exploration
    ):
        super().__init__()
        
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self.success_distance = success_distance
        self.rotation_step = rotation_step
        self.movement_step = movement_step
        self.context_size = context_size
        self.goal_prob = goal_prob
        
        # Default scenes if none provided
        if scene_names is None:
            self.scene_names = [
                'FloorPlan1', 'FloorPlan2', 'FloorPlan3', 'FloorPlan4', 'FloorPlan5',
                'FloorPlan201', 'FloorPlan202', 'FloorPlan203', 'FloorPlan204', 'FloorPlan205',
                'FloorPlan301', 'FloorPlan302', 'FloorPlan303', 'FloorPlan304', 'FloorPlan305',
                'FloorPlan401', 'FloorPlan402', 'FloorPlan403', 'FloorPlan404', 'FloorPlan405',
            ]
        else:
            self.scene_names = scene_names
        
        # Initialize AI2Thor controller
        self.controller = Controller(
            agentMode="default",
            visibilityDistance=1.5,
            scene=random.choice(self.scene_names),
            gridSize=0.25,
            snapToGrid=True,
            rotateStepDegrees=self.rotation_step,
            renderDepthImage=True,
            renderInstanceSegmentation=True,
            width=224,
            height=224,
            fieldOfView=90
        )
        
        # Action space: [move_forward, move_backward, turn_left, turn_right, look_up, look_down]
        self.action_space = spaces.Discrete(6)
        
        # Observation space: RGB images + goal + context + goal_mask
        self.observation_space = spaces.Dict({
            'rgb': spaces.Box(low=0, high=255, shape=(3, *image_size), dtype=np.uint8),
            'goal_rgb': spaces.Box(low=0, high=255, shape=(3, *image_size), dtype=np.uint8),
            'context': spaces.Box(low=0, high=255, shape=(3 * context_size, *image_size), dtype=np.uint8),
            'goal_mask': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'goal_position': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })
        
        # Internal state
        self.current_scene = None
        self.current_step = 0
        self.context_buffer = []
        self.goal_position = None
        self.goal_image = None
        self.is_goal_conditioned = False
        self.initial_position = None
        self.visited_positions = set()
        self.object_positions = {}
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return initial observation"""
        # Randomly select scene
        scene_name = random.choice(self.scene_names)
        if scene_name != self.current_scene:
            self.controller.reset(scene=scene_name)
            self.current_scene = scene_name
            self._cache_object_positions()
        
        # Reset agent to random position
        reachable_positions = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        if reachable_positions:
            start_pos = random.choice(reachable_positions)
            self.controller.step(
                action="Teleport",
                position=start_pos,
                rotation=dict(x=0, y=random.choice([0, 90, 180, 270]), z=0)
            )
        
        # Reset internal state
        self.current_step = 0
        self.context_buffer = []
        self.visited_positions = set()
        self.initial_position = self._get_agent_position()
        
        # Decide if this episode is goal-conditioned or exploration
        self.is_goal_conditioned = random.random() < self.goal_prob
        
        if self.is_goal_conditioned:
            self._set_random_goal()
        else:
            self.goal_position = None
            self.goal_image = None
        
        # Get initial observation
        obs = self._get_observation()
        
        # Initialize context buffer
        for _ in range(self.context_size):
            self.context_buffer.append(obs['rgb'])
        
        return self._format_observation(obs)
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Execute action and return next observation, reward, done, info"""
        # Map discrete action to AI2Thor action
        action_map = {
            0: "MoveAhead",
            1: "MoveBack", 
            2: "RotateLeft",
            3: "RotateRight",
            4: "LookUp",
            5: "LookDown"
        }
        
        # Execute action
        event = self.controller.step(action=action_map[action])
        self.current_step += 1
        
        # Get new observation
        obs = self._get_observation()
        
        # Update context buffer
        self.context_buffer.append(obs['rgb'])
        if len(self.context_buffer) > self.context_size:
            self.context_buffer.pop(0)
        
        # Calculate reward
        reward = self._calculate_reward(event, obs)
        
        # Check if episode is done
        done = self._is_done(obs)
        
        # Additional info
        info = {
            'success': self._is_success(obs),
            'collision': not event.metadata['lastActionSuccess'],
            'step': self.current_step,
            'goal_conditioned': self.is_goal_conditioned,
            'distance_to_goal': self._distance_to_goal() if self.is_goal_conditioned else 0.0
        }
        
        return self._format_observation(obs), reward, done, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation from AI2Thor"""
        event = self.controller.last_event
        
        # Get RGB image
        rgb_image = np.array(event.frame)
        rgb_image = cv2.resize(rgb_image, self.image_size)
        rgb_image = np.transpose(rgb_image, (2, 0, 1))  # CHW format
        
        # Get goal image (if goal-conditioned)
        if self.goal_image is not None:
            goal_rgb = np.array(self.goal_image)
            goal_rgb = cv2.resize(goal_rgb, self.image_size)
            goal_rgb = np.transpose(goal_rgb, (2, 0, 1))
        else:
            goal_rgb = np.zeros((3, *self.image_size), dtype=np.uint8)
        
        return {
            'rgb': rgb_image,
            'goal_rgb': goal_rgb,
            'position': np.array([
                event.metadata['agent']['position']['x'],
                event.metadata['agent']['position']['y'], 
                event.metadata['agent']['position']['z']
            ]),
            'rotation': event.metadata['agent']['rotation']['y']
        }
    
    def _format_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Format observation for the model"""
        # Stack context images
        if len(self.context_buffer) == self.context_size:
            context = np.concatenate(self.context_buffer, axis=0)
        else:
            # Pad with zeros if not enough context
            context = np.concatenate(
                self.context_buffer + [np.zeros_like(obs['rgb'])] * (self.context_size - len(self.context_buffer)),
                axis=0
            )
        
        return {
            'rgb': obs['rgb'].astype(np.uint8),
            'goal_rgb': obs['goal_rgb'].astype(np.uint8),
            'context': context.astype(np.uint8),
            'goal_mask': np.array([0.0 if self.is_goal_conditioned else 1.0], dtype=np.float32),
            'goal_position': self.goal_position if self.goal_position is not None else np.zeros(3, dtype=np.float32),
        }
    
    def _set_random_goal(self):
        """Set a random navigation goal"""
        # Get reachable positions
        reachable_positions = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        
        if reachable_positions:
            # Choose a goal position far from start
            current_pos = self._get_agent_position()
            valid_goals = [
                pos for pos in reachable_positions 
                if self._calculate_distance(current_pos, pos) > 2.0  # At least 2 meters away
            ]
            
            if valid_goals:
                goal_pos = random.choice(valid_goals)
                self.goal_position = np.array([goal_pos['x'], goal_pos['y'], goal_pos['z']])
                
                # Navigate to goal position to capture goal image
                current_state = self.controller.last_event.metadata['agent']
                self.controller.step(
                    action="Teleport",
                    position=goal_pos,
                    rotation=dict(x=0, y=random.choice([0, 90, 180, 270]), z=0)
                )
                goal_event = self.controller.last_event
                self.goal_image = goal_event.frame
                
                # Return to original position
                self.controller.step(
                    action="Teleport",
                    position=current_state['position'],
                    rotation=current_state['rotation']
                )
    
    def _calculate_reward(self, event, obs) -> float:
        """Calculate reward for the current step"""
        reward = 0.0
        
        # Basic step penalty to encourage efficiency
        reward -= 0.01
        
        if self.is_goal_conditioned:
            # Goal-conditioned rewards
            distance = self._distance_to_goal()
            
            # Dense distance reward
            if hasattr(self, '_prev_distance_to_goal'):
                distance_improvement = self._prev_distance_to_goal - distance
                reward += distance_improvement * 10.0
            
            self._prev_distance_to_goal = distance
            
            # Success reward
            if distance < self.success_distance:
                reward += 100.0
            
        else:
            # Exploration rewards
            current_pos = self._get_agent_position()
            pos_key = (round(current_pos['x'], 1), round(current_pos['z'], 1))
            
            # Reward for visiting new areas
            if pos_key not in self.visited_positions:
                reward += 5.0
                self.visited_positions.add(pos_key)
            
            # Small reward for forward movement (encourages exploration)
            if event.metadata['lastAction'] == 'MoveAhead' and event.metadata['lastActionSuccess']:
                reward += 0.1
        
        # Collision penalty
        if not event.metadata['lastActionSuccess']:
            reward -= 5.0
        
        return reward
    
    def _is_done(self, obs) -> bool:
        """Check if episode should end"""
        if self.current_step >= self.max_episode_steps:
            return True
        
        if self.is_goal_conditioned:
            return self._distance_to_goal() < self.success_distance
        
        return False
    
    def _is_success(self, obs) -> bool:
        if self.is_goal_conditioned:
            return self._distance_to_goal() < self.success_distance
        else:
            # For exploration, success is visiting many unique locations
            return len(self.visited_positions) > 20
    
    def _distance_to_goal(self) -> float:
        """Calculate distance to goal"""
        if self.goal_position is None:
            return float('inf')
        
        current_pos = self._get_agent_position()
        return self._calculate_distance(current_pos, {
            'x': self.goal_position[0],
            'y': self.goal_position[1], 
            'z': self.goal_position[2]
        })
    
    def _get_agent_position(self) -> Dict[str, float]:
        """Get current agent position"""
        return self.controller.last_event.metadata['agent']['position']
    
    def _calculate_distance(self, pos1, pos2) -> float:
        return np.sqrt(
            (pos1['x'] - pos2['x'])**2 + 
            (pos1['z'] - pos2['z'])**2
        )
    
    def _cache_object_positions(self):
        event = self.controller.last_event
        self.object_positions = {}
        
        for obj in event.metadata['objects']:
            if obj['pickupable'] or obj['objectType'] in ['Television', 'Microwave', 'Fridge']:
                self.object_positions[obj['objectId']] = obj['position']
    
    def close(self):
        if self.controller:
            self.controller.stop()
    
    def render(self, mode='human'):
        if mode == 'human':
            event = self.controller.last_event
            return np.array(event.frame)
        return None