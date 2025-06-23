import os
import torch
import numpy as np
import yaml
from ai2thor_nomad.ai2thor_env import AI2ThorNoMaDEnv
from deployment.src.utils import load_model, transform_images, to_numpy
from vint_train.training.train_utils import get_action
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import argparse

def evaluate_nomad_ai2thor(model_name="nomad", start_floor=1, end_floor=430):
    MODEL_CONFIG_PATH = "deployment/config/models.yaml"
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)
    
    model_config_path = model_paths[model_name]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpth_path = model_paths[model_name]["ckpt_path"]
    
    if not os.path.exists(ckpth_path):
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    
    model = load_model(ckpth_path, model_params, device)
    model = model.to(device)
    model.eval()
    
    # Setup noise scheduler for diffusion
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    
    # Initialize AI2Thor environment 
    ai2thor_env = AI2ThorNoMaDEnv()
    
    results = {}
    
    for floor_plan in range(start_floor, end_floor + 1):
        print(f"Evaluating on FloorPlan{floor_plan}")
        
        try:
            # Reset environment to specific floor plan
            scene_name = f"FloorPlan{floor_plan}"
            ai2thor_env.reset(scene=scene_name)
            
            # Get initial observation
            obs = ai2thor_env.get_observation()
            context_queue = [obs]
            
            success_rate = 0
            num_episodes = 10  # Number of episodes per floor plan
            
            for episode in range(num_episodes):
                episode_success = evaluate_episode(
                    ai2thor_env, model, noise_scheduler, 
                    model_params, context_queue, device
                )
                success_rate += episode_success
            
            success_rate /= num_episodes
            results[floor_plan] = success_rate
            
            print(f"FloorPlan{floor_plan}: Success Rate = {success_rate:.3f}")
            
        except Exception as e:
            print(f"Error on FloorPlan{floor_plan}: {e}")
            results[floor_plan] = 0.0
    
    # Save results
    with open("ai2thor_evaluation_results.yaml", "w") as f:
        yaml.dump(results, f)
    
    overall_success = np.mean(list(results.values()))
    print(f"\nOverall Success Rate: {overall_success:.3f}")
    
    return results

def evaluate_episode(env, model, noise_scheduler, model_params, context_queue, device):
    """Evaluate a single episode"""
    max_steps = 500
    success = False
    
    for step in range(max_steps):
        # Get current observation
        obs = env.get_observation()
        context_queue.append(obs)
        
        # Maintain context window
        if len(context_queue) > model_params["context_size"] + 1:
            context_queue.pop(0)
        
        if len(context_queue) > model_params["context_size"]:
            # Transform observations
            obs_images = transform_images(
                context_queue[-model_params["context_size"]:], 
                model_params["image_size"], 
                center_crop=False
            ).to(device)
            
            # Create fake goal for exploration
            fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)
            mask = torch.ones(1).long().to(device)  # Mask goal for exploration
            
            # Get action from model
            with torch.no_grad():
                obs_cond = model('vision_encoder', 
                               obs_img=obs_images, 
                               goal_img=fake_goal, 
                               input_goal_mask=mask)
                
                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(8, 1)  # num_samples
                else:
                    obs_cond = obs_cond.repeat(8, 1, 1)
                
                # Sample action using diffusion
                noisy_action = torch.randn(
                    (8, model_params["len_traj_pred"], 2), device=device)
                naction = noisy_action
                
                noise_scheduler.set_timesteps(model_params["num_diffusion_iters"])
                
                for k in noise_scheduler.timesteps:
                    noise_pred = model(
                        'noise_pred_net',
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
                
                naction = to_numpy(get_action(naction))
                chosen_waypoint = naction[0][2]  # Choose middle waypoint
            
            # Execute action in environment
            action_dict = waypoint_to_action(chosen_waypoint)
            obs, reward, done, info = env.step(action_dict)
            
            # Check for success condition (define based on your task)
            if done and reward > 0:
                success = True
                break
    
    return success

def waypoint_to_action(waypoint):
    x, y = waypoint
    
    if abs(x) > abs(y):
        if x > 0:
            return {"action": "MoveRight"}
        else:
            return {"action": "MoveLeft"}
    else:
        if y > 0:
            return {"action": "MoveAhead"}
        else:
            return {"action": "MoveBack"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nomad", help="Model name")
    parser.add_argument("--start_floor", type=int, default=1, help="Start floor plan")
    parser.add_argument("--end_floor", type=int, default=430, help="End floor plan")
    
    args = parser.parse_args()
    
    evaluate_nomad_ai2thor(args.model, args.start_floor, args.end_floor)