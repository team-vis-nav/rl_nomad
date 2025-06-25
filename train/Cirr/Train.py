import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import argparse
import yaml
import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
import sys
sys.path.append(r'/home/tuandang/tuandang/quanganh/visualnav-transformer/train')

from Unified.env import EnhancedAI2ThorEnv
from TwoStage.nomad_model import (
    EnhancedNoMaDRL, MultiComponentRewardCalculator, 
    CurriculumManager, EvaluationMetrics, prepare_observation
)
from TwoStage.two_stage_train import TwoStageTrainer

sys.path.append(r'/home/tuandang/tuandang/quanganh/visualnav-transformer/train/Cirr')
from Cirr import EnhancedCurriculumManager

class UnifiedTrainerWithCurriculum(TwoStageTrainer):
    def __init__(self, config: Dict):
        self.dataset = config['dataset']
        self.splits = self._load_splits(config)
        
        self.curriculum_manager = EnhancedCurriculumManager(config, self.dataset)
        curriculum_settings = self.curriculum_manager.get_current_settings()
        
        self.all_train_scenes = self.splits['train']
        config['scene_names'] = self.curriculum_manager.get_current_scenes(self.all_train_scenes)
        
        config.update({
            'max_episode_steps': curriculum_settings['max_episode_steps'],
            'goal_prob': curriculum_settings['goal_prob']
        })
        
        super().__init__(config)
        
        self.reward_calculator = self._create_curriculum_reward_calculator(config)
        
        self.val_scenes = self.splits['val']
        self.test_scenes = self.splits['test']

        # self.val_env = self._create_environment(
        #     self.splits['val'], 
        #     config, 
        #     goal_prob=config.get('eval_goal_prob', 1.0)
        # )
        
        # self.test_env = self._create_environment(
        #     self.splits['test'], 
        #     config, 
        #     goal_prob=config.get('eval_goal_prob', 1.0)
        # )
        
        self.results = {
            'train': defaultdict(list),
            'val': defaultdict(list),
            'test': defaultdict(list),
            'curriculum': defaultdict(list)
        }
        
        self.best_val_success_rate = 0
        self.best_val_checkpoint = None
        
        self.last_curriculum_update = 0
        self.curriculum_update_freq = config.get('curriculum_update_freq', 10)

        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.get('learning_rate', 1e-5),  # Very low
            weight_decay=0.01,
            eps=1e-5
        )

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.get('learning_rate', 1e-5) * 10,
            total_steps=config['stage1_timesteps'] // config['rollout_steps'],
            pct_start=0.1,
            anneal_strategy='linear'
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param, gain=0.1)
                elif 'policy' in name and param.dim() == 2 and param.size(0) == 4:
                    # Output layer of policy
                    nn.init.normal_(param, mean=0.0, std=0.001)
                elif param.dim() == 2:
                    nn.init.xavier_normal_(param, gain=0.5)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def _create_curriculum_reward_calculator(self, config: Dict) -> MultiComponentRewardCalculator:
        """Create reward calculator with curriculum-adjusted settings"""
        curriculum_settings = self.curriculum_manager.get_current_settings()
        
        # Adjust reward config based on curriculum
        reward_config = config.copy()
        reward_config['training_stage'] = self.stage
        
        # Apply curriculum-specific multipliers
        if 'collision_penalty_multiplier' in curriculum_settings:
            reward_config['collision_penalty'] = (
                config.get('collision_penalty', 1.0) * 
                curriculum_settings['collision_penalty_multiplier']
            )
        
        return MultiComponentRewardCalculator(reward_config)
    
    def _create_environment(self, scenes: List[str], config: Dict, goal_prob: float):
        return EnhancedAI2ThorEnv(
            scene_names=scenes,
            image_size=tuple(config['image_size']),
            max_episode_steps=config['max_episode_steps'],
            success_distance=config['success_distance'],
            context_size=config['context_size'],
            goal_prob=goal_prob
        )
    
    def _update_environment_curriculum(self):
        curriculum_settings = self.curriculum_manager.get_current_settings()
        
        new_scenes = self.curriculum_manager.get_current_scenes(self.all_train_scenes)
        
        self.env.close()
        self.env = self._create_environment(
            new_scenes,
            self.config,
            goal_prob=curriculum_settings['goal_prob']
        )
        
        self.env.max_episode_steps = curriculum_settings['max_episode_steps']
        self.env.goal_prob = curriculum_settings['goal_prob']
        
        # Update reward calculator
        self.reward_calculator = self._create_curriculum_reward_calculator(self.config)
        
        # Update success distance based on curriculum
        if curriculum_settings['max_distance'] != float('inf'):
            # Temporarily constrain goal selection distance
            self.env.max_goal_distance = curriculum_settings['max_distance']
        
        print(f"Updated environment with {len(new_scenes)} scenes for curriculum level {self.curriculum_manager.current_level}")
    
    def collect_rollouts(self, num_steps: int) -> Dict[str, float]:
        obs = self.env.reset()
        torch_obs = prepare_observation(obs, self.device)
        print("torch obs: ", torch_obs['rgb'].shape)
        save_image(torch_obs['rgb'].squeeze(), 'log_img/output_image0.png')
        
        # if self.debug_mode:
        #     for key, val in torch_obs.items():
        #         print(f"Initial obs - {key}: shape={val.shape}")
        
        self.model.reset_hidden()
        hidden_state = None
        
        episode_reward = 0
        episode_length = 0
        episode_success = False
        episode_collisions = 0
        
        rollout_stats = {
            'total_reward': 0,
            'episodes': 0,
            'successes': 0,
            'collisions': 0,
            'auxiliary_loss': 0
        }
        
        for step in range(num_steps):
            with torch.no_grad():
                for key in torch_obs:
                    if torch_obs[key].dim() == 3 and key in ['rgb', 'goal_rgb', 'context']:
                        if torch_obs[key].size(0) != 1:  # Not already batched
                            torch_obs[key] = torch_obs[key].unsqueeze(0)
                    elif torch_obs[key].dim() == 1:
                        torch_obs[key] = torch_obs[key].unsqueeze(0)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model.forward(
                            torch_obs, mode="all", hidden_state=hidden_state
                        )
                else:
                    outputs = self.model.forward(
                        torch_obs, mode="all", hidden_state=hidden_state
                    )
                
                action_dist = outputs['action_dist']
                value = outputs['values']
                hidden_state = outputs['hidden_state']
                
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            
            next_obs, _, done, info = self.env.step(action.cpu().item())
            
            # Calculate reward
            event = self.env.controller.last_event
            reward = self.reward_calculator.calculate_reward(event, next_obs, info, self.env)
            
            next_torch_obs = prepare_observation(next_obs, self.device)
            save_image(next_torch_obs['rgb'].squeeze(), f"log_img/output_image{step}.png")
            print("torch obs: ", next_torch_obs['rgb'].shape)
            # Store transition (remove batch dimension for storage)
            store_obs = {}
            for key, val in torch_obs.items():
                if val.dim() > 0 and val.size(0) == 1:
                    store_obs[key] = val.squeeze(0)
                else:
                    store_obs[key] = val
            
            self.buffer.store(
                obs=store_obs,
                action=action.squeeze(0) if action.dim() > 0 else action,
                reward=reward,
                value=value.squeeze(0) if value.dim() > 0 else value,
                log_prob=log_prob.squeeze(0) if log_prob.dim() > 0 else log_prob,
                done=done
            )
            
            episode_reward += reward
            episode_length += 1
            if info.get('collision', False):
                episode_collisions += 1
            if info.get('success', False):
                episode_success = True
            
            rollout_stats['total_reward'] += reward
            
            # Episode termination
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.success_rates.append(1.0 if episode_success else 0.0)
                
                # Update curriculum
                self.curriculum_manager.update(
                    episode_success, 
                    episode_length,
                    episode_collisions
                )
                
                rollout_stats['episodes'] += 1
                rollout_stats['successes'] += 1 if episode_success else 0
                rollout_stats['collisions'] += episode_collisions
                
                # Reset for new episode
                obs = self.env.reset()
                torch_obs = prepare_observation(obs, self.device)
                print("torch obs: ", torch_obs['rgb'].shape)
                save_image(torch_obs['rgb'].squeeze(), f"log_img/output_image{step}.png")
                self.model.reset_hidden()
                hidden_state = None
                episode_reward = 0
                episode_length = 0
                episode_success = False
                episode_collisions = 0
            else:
                torch_obs = next_torch_obs
        
        # Compute GAE
        with torch.no_grad():
            # Ensure proper batch dimension for final value computation
            for key in torch_obs:
                if torch_obs[key].dim() == 3 and key in ['rgb', 'goal_rgb', 'context']:
                    if torch_obs[key].size(0) != 1:
                        torch_obs[key] = torch_obs[key].unsqueeze(0)
                elif torch_obs[key].dim() == 1:
                    torch_obs[key] = torch_obs[key].unsqueeze(0)
            
            if self.use_amp:
                with autocast():
                    final_value = self.model.forward(
                        torch_obs, mode="value", hidden_state=hidden_state
                    )['values'].cpu().item()
            else:
                final_value = self.model.forward(
                    torch_obs, mode="value", hidden_state=hidden_state
                )['values'].cpu().item()
        
        self.buffer.compute_gae(final_value, self.gamma, self.lam)
        
        return rollout_stats
    
    def _evaluate_on_split(self, split_name: str, env=None, num_episodes: Optional[int] = None) -> Dict[str, float]:
        if num_episodes is None:
            num_episodes = self.config.get('eval_episodes', 20)
        
        print(f"\nEvaluating on {split_name} split ({num_episodes} episodes)...")
        
        original_scenes = self.env.scene_names
        original_goal_prob = self.env.goal_prob
        
        if split_name == 'val':
            self.env.scene_names = self.val_scenes
        elif split_name == 'test':
            self.env.scene_names = self.test_scenes
        elif split_name == 'train':
            self.env.scene_names = self.splits['train']  # Use full training set
        
        self.env.goal_prob = self.config.get('eval_goal_prob', 1.0)
        
        self.model.eval()
        
        metrics = EvaluationMetrics()
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        episode_collisions = []
        
        for episode_idx in range(num_episodes):
            obs = self.env.reset()
            torch_obs = prepare_observation(obs, self.device)
            
            if hasattr(self.model, 'reset_hidden'):
                self.model.reset_hidden()
            hidden_state = None
            
            episode_reward = 0
            episode_length = 0
            collision_count = 0
            
            while episode_length < self.config['max_episode_steps']:
                with torch.no_grad():
                    if hasattr(self.model, 'get_action'):
                        action, _, hidden_state = self.model.get_action(
                            torch_obs, deterministic=True, hidden_state=hidden_state
                        )
                    else:
                        outputs = self.model.forward(torch_obs, mode="policy")
                        action_dist = outputs['action_dist']
                        action = action_dist.probs.argmax(dim=-1)
                
                next_obs, _, done, info = self.env.step(action.cpu().item())
                
                event = self.env.controller.last_event
                reward = self.reward_calculator.calculate_reward(event, next_obs, info, self.env)
                
                episode_reward += reward
                episode_length += 1
                
                if info.get('collision', False):
                    collision_count += 1
                
                agent_pos = self.env._get_agent_position()
                pos_key = (round(agent_pos['x'], 1), round(agent_pos['z'], 1))
                metrics.step(info, reward, pos_key)
                
                if done:
                    break
                
                torch_obs = prepare_observation(next_obs, self.device)
            
            episode_success = info.get('success', False)
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_successes.append(episode_success)
            episode_collisions.append(collision_count)
            metrics.end_episode(episode_success)
            
            if (episode_idx + 1) % 10 == 0:
                current_sr = sum(episode_successes) / len(episode_successes)
                print(f"  Completed {episode_idx + 1}/{num_episodes} episodes (SR: {current_sr:.2%})")
        
        self.env.scene_names = original_scenes
        self.env.goal_prob = original_goal_prob
        self.model.train()
        
        eval_metrics = metrics.compute_metrics()
        eval_metrics.update({
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': np.mean(episode_successes),
            'avg_collisions': np.mean(episode_collisions),
            'num_episodes': num_episodes
        })
        
        print(f"\n{split_name.upper()} Results:")
        print(f"  Success Rate: {eval_metrics['success_rate']:.2%}")
        print(f"  Avg Reward: {eval_metrics['avg_reward']:.2f}")
        print(f"  Avg Length: {eval_metrics['avg_length']:.1f}")
        
        return eval_metrics

    def train_val_test(self, total_timesteps: int):
        print(f"Starting unified training with curriculum learning for {total_timesteps} timesteps...")
        print(f"Dataset: {self.dataset}")
        print(f"Stage: {self.stage}")
        print(f"Using single environment instance to avoid Unity conflicts")
        
        timesteps_collected = 0
        update_count = 0
        
        while timesteps_collected < total_timesteps:
            rollout_stats = self.collect_rollouts(self.config['rollout_steps'])
            timesteps_collected += self.config['rollout_steps']
            
            update_stats = self.update_policy()
            update_count += 1
            
            if update_count % 10 == 0:
                torch.cuda.empty_cache()
            
            if update_count % self.config['log_freq'] == 0:
                self._log_training_stats_with_curriculum(
                    timesteps_collected, rollout_stats, update_stats
                )
            
            if update_count % self.config.get('val_freq', 100) == 0:
                val_metrics = self._evaluate_on_split('val')
                self.results['val']['timesteps'].append(timesteps_collected)
                for key, value in val_metrics.items():
                    self.results['val'][key].append(value)
                
                if val_metrics['success_rate'] > self.best_val_success_rate:
                    self.best_val_success_rate = val_metrics['success_rate']
                    self.best_val_checkpoint = self._save_model(
                        update_count, timesteps_collected, is_best=True
                    )
                    print(f"New best validation success rate: {self.best_val_success_rate:.2%}")
            
            if update_count % self.config['save_freq'] == 0:
                self._save_model(update_count, timesteps_collected)
        
        print("\nTraining completed! Running final evaluation...")
        
        if self.best_val_checkpoint and os.path.exists(self.best_val_checkpoint):
            print(f"Loading best model from {self.best_val_checkpoint}")
            checkpoint = torch.load(self.best_val_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        final_results = {}
        
        print("\n=== Training Set Evaluation ===")
        train_metrics = self._evaluate_on_split('train', num_episodes=50)
        final_results['train'] = train_metrics
        
        print("\n=== Validation Set Evaluation ===")
        val_metrics = self._evaluate_on_split('val', num_episodes=50)
        final_results['val'] = val_metrics
        
        print("\n=== Test Set Evaluation ===")
        test_metrics = self._evaluate_on_split('test', num_episodes=100)
        final_results['test'] = test_metrics
        
        self._save_final_results_with_curriculum(final_results)
        self.env.close()
        
        return final_results
    
    def _log_training_stats_with_curriculum(self, timesteps: int, rollout_stats: Dict, update_stats: Dict):
        curriculum_stats = self.curriculum_manager.get_progress_stats()
        
        print(f"\n--- Stage {self.stage} Update {timesteps // self.config['rollout_steps']} "
              f"(Timesteps: {timesteps}) ---")
        print(f"Curriculum Level: {curriculum_stats['current_level']} ({curriculum_stats['level_name']})")
        print(f"Episodes at Level: {curriculum_stats['episodes_at_level']}")
        print(f"Progress: {curriculum_stats['levels_completed']}/{curriculum_stats['total_levels']} levels")
        
        if len(self.episode_rewards) > 0:
            print(f"Episode Reward: {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}")
            print(f"Episode Length: {np.mean(self.episode_lengths):.1f}")
            print(f"Success Rate: {np.mean(self.success_rates):.2%}")
            
            # Compute comprehensive metrics
            metrics = self.evaluation_metrics.compute_metrics()
            if metrics:
                print(f"SPL: {metrics.get('spl', 0):.3f}")
                print(f"Collision-Free Success Rate: {metrics.get('collision_free_success_rate', 0):.2%}")
                print(f"Exploration Coverage: {metrics.get('exploration_coverage', 0):.3f}")
        
        print(f"Policy Loss: {update_stats['policy_loss']:.4f}")
        print(f"Value Loss: {update_stats['value_loss']:.4f}")
        print(f"Entropy: {-update_stats['entropy_loss']:.4f}")
        
        if self.stage == 1 and update_stats['auxiliary_loss'] > 0:
            print(f"Auxiliary Loss: {update_stats['auxiliary_loss']:.4f}")
        
        print(f"Approx KL: {update_stats['approx_kl']:.4f}")
        
        if self.config.get('use_wandb', False):
            log_dict = {
                'timesteps': timesteps,
                'stage': self.stage,
                'curriculum/level': curriculum_stats['current_level'],
                'curriculum/level_name': curriculum_stats['level_name'],
                'curriculum/episodes_at_level': curriculum_stats['episodes_at_level'],
                'curriculum/level_success_rate': curriculum_stats['current_success_rate'],
                'episode_reward_mean': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'episode_length_mean': np.mean(self.episode_lengths) if self.episode_lengths else 0,
                'success_rate': np.mean(self.success_rates) if self.success_rates else 0,
                'policy_loss': update_stats['policy_loss'],
                'value_loss': update_stats['value_loss'],
                'entropy': -update_stats['entropy_loss'],
                'distance_loss': update_stats['distance_loss'],
                'approx_kl': update_stats['approx_kl'],
                'clip_fraction': update_stats['clip_fraction'],
            }
            
            if self.stage == 1:
                log_dict['auxiliary_loss'] = update_stats['auxiliary_loss']
            
            metrics = self.evaluation_metrics.compute_metrics()
            if metrics:
                log_dict.update({f'train/{k}': v for k, v in metrics.items()})
            
            wandb.log(log_dict)
    
    def _run_final_evaluation(self) -> Dict:
        final_results = {}
        
        print("\n=== Training Set Evaluation ===")
        self.env.close()
        self.env = self._create_environment(
            self.splits['train'], 
            self.config,
            goal_prob=self.config.get('eval_goal_prob', 1.0)
        )
        train_metrics = self._evaluate_on_split('train', self.env, num_episodes=50)
        final_results['train'] = train_metrics
        
        print("\n=== Validation Set Evaluation ===")
        val_metrics = self._evaluate_on_split('val', self.val_env, num_episodes=50)
        final_results['val'] = val_metrics
        
        print("\n=== Test Set Evaluation ===")
        test_metrics = self._evaluate_on_split('test', self.test_env, num_episodes=100)
        final_results['test'] = test_metrics
        
        return final_results
    
    def _save_final_results_with_curriculum(self, results: Dict):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(self.config['save_dir'], 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        curriculum_history = self.curriculum_manager.performance_history
        results_file = os.path.join(
            results_dir, 
            f'{self.dataset}_stage{self.stage}_{timestamp}_results.json'
        )
        
        with open(results_file, 'w') as f:
            json.dump({
                'config': self.config,
                'final_results': results,
                'training_history': {
                    split: dict(self.results[split]) 
                    for split in ['train', 'val', 'test', 'curriculum']
                },
                'curriculum_progression': {
                    'final_level': self.curriculum_manager.current_level,
                    'levels_completed': self.curriculum_manager.current_level,
                    'total_levels': len(self.curriculum_manager.levels),
                    'performance_history': curriculum_history
                },
                'best_val_success_rate': self.best_val_success_rate,
                'timestamp': timestamp
            }, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        print("\n" + "="*70)
        print("FINAL RESULTS SUMMARY")
        print("="*70)
        print(f"Final Curriculum Level: {self.curriculum_manager.current_level} / {len(self.curriculum_manager.levels) - 1}")
        print(f"Curriculum Progression: {self.curriculum_manager.get_current_settings()['name']}")
        print("-"*70)
        print(f"{'Metric':<25} {'Train':>10} {'Val':>10} {'Test':>10}")
        print("-"*70)
        
        metrics_to_show = [
            ('Success Rate (%)', 'success_rate', 100),
            ('SPL', 'spl', 1),
            ('Avg Reward', 'avg_reward', 1),
            ('Avg Episode Length', 'avg_length', 1),
            ('Collision-Free SR (%)', 'collision_free_success_rate', 100)
        ]
        
        for metric_name, metric_key, multiplier in metrics_to_show:
            train_val = results['train'].get(metric_key, 0) * multiplier
            val_val = results['val'].get(metric_key, 0) * multiplier
            test_val = results['test'].get(metric_key, 0) * multiplier
            print(f"{metric_name:<25} {train_val:>10.2f} {val_val:>10.2f} {test_val:>10.2f}")
        
        print("="*70)
        
        if curriculum_history:
            print("\n" + "="*70)
            print("CURRICULUM PROGRESSION")
            print("="*70)
            print(f"{'Level':<6} {'Name':<20} {'Episodes':<10} {'Success':<10} {'Collisions':<12}")
            print("-"*70)
            
            for i, hist in enumerate(curriculum_history):
                level_name = self.curriculum_manager.levels[hist['level']]['name']
                print(f"{hist['level']:<6} {level_name:<20} {hist['episodes']:<10} "
                      f"{hist['success_rate']*100:<10.1f} {hist['avg_collisions']:<12.2f}")
            
            print("="*70)
    
    def _load_splits(self, config: Dict) -> Dict[str, List[str]]:
        dataset = config['dataset']
        
        if dataset == 'combined':
            splits_file = config.get('splits_file', '/home/tuandang/tuandang/quanganh/visualnav-transformer/train/Unified/config/splits/combined_splits.yaml')
        else:
            splits_file = config.get('splits_file', f'/home/tuandang/tuandang/quanganh/visualnav-transformer/train/Unified/config/splits/{dataset}_splits.yaml')
        
        if os.path.exists(splits_file):
            with open(splits_file, 'r') as f:
                splits = yaml.safe_load(f)
            print(f"Loaded splits from {splits_file}")
        else:
            if dataset == 'combined':
                from Unified.datasplit import CombinedAI2THORDatasetSplitter
                splitter = CombinedAI2THORDatasetSplitter()
                splits_dict = splitter.save_combined_splits()
                splits = {k: v['combined'] for k, v in splits_dict.items()}
            else:
                from TwoStage.datasplit import AI2THORDatasetSplitter
                splitter = AI2THORDatasetSplitter()
                splits = splitter.save_splits(dataset)
        
        print(f"Dataset: {dataset}")
        print(f"Train scenes: {len(splits['train'])}")
        print(f"Val scenes: {len(splits['val'])}")
        print(f"Test scenes: {len(splits['test'])}")
        
        return splits
    
    def _save_model(self, update_count: int, timesteps: int, is_best: bool = False):
        """Save model checkpoint with curriculum information"""
        curriculum_stats = self.curriculum_manager.get_progress_stats()
        
        checkpoint_name = f'{self.dataset}_stage{self.stage}_{"best" if is_best else update_count}.pth'
        save_path = os.path.join(self.config['save_dir'], checkpoint_name)
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': update_count,
            'timesteps': timesteps,
            'stage': self.stage,
            'curriculum_level': curriculum_stats['current_level'],
            'curriculum_stats': curriculum_stats,
            'config': self.config,
            'success_rate': np.mean(self.success_rates) if self.success_rates else 0,
        }, save_path)
        
        print(f"Model saved to {save_path}")
        return save_path


def main():
    parser = argparse.ArgumentParser(description='Unified Train/Val/Test with Curriculum Learning')
    parser.add_argument('--config', type=str, required=False, default="/home/tuandang/tuandang/quanganh/visualnav-transformer/train/config/cirr.yaml",
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, choices=['ithor', 'robothor', 'combined'],
                       required=True, help='Dataset to use')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=1,
                       help='Training stage')
    parser.add_argument('--stage1-checkpoint', type=str, default=None,
                       help='Stage 1 checkpoint for stage 2')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'test'],
                       default='train', help='Mode to run')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint for evaluation/testing')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['dataset'] = args.dataset
    config['training_stage'] = args.stage
    if args.stage1_checkpoint:
        config['stage1_checkpoint'] = args.stage1_checkpoint
    
    if config.get('use_wandb', False) and args.mode == 'train':
        wandb.init(
            project=config.get('wandb_project', 'nomad-rl-curriculum'),
            name=f"{args.dataset}_{config.get('run_name', 'curriculum')}_stage{args.stage}",
            config=config
        )
    
    trainer = UnifiedTrainerWithCurriculum(config)
    
    if args.mode == 'train':
        results = trainer.train_val_test(config[f'stage{args.stage}_timesteps'])
    
    elif args.mode == 'eval':
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from curriculum level: {checkpoint.get('curriculum_level', 'unknown')}")
        
        val_metrics = trainer._evaluate_on_split('val', trainer.val_env, num_episodes=100)
        print("\nValidation metrics:", val_metrics)
    
    elif args.mode == 'test':
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from curriculum level: {checkpoint.get('curriculum_level', 'unknown')}")
        
        test_metrics = trainer._evaluate_on_split('test', trainer.test_env, num_episodes=200)
        print("\nTest metrics:", test_metrics)

if __name__ == '__main__':
    main()
