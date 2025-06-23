import random
import yaml
import os
from typing import Dict, List, Tuple
import numpy as np

class AI2THORDatasetSplitter:    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.ithor_scenes = {
            'kitchen': [f'FloorPlan{i}' for i in range(1, 31)],
            'living_room': [f'FloorPlan{i}' for i in range(201, 231)],
            'bedroom': [f'FloorPlan{i}' for i in range(301, 331)],
            'bathroom': [f'FloorPlan{i}' for i in range(401, 431)]
        }
        
        self.robothor_splits = {
            'train': [
                'FloorPlan_Train1_1', 'FloorPlan_Train1_2', 'FloorPlan_Train1_3', 'FloorPlan_Train1_4', 'FloorPlan_Train1_5',
                'FloorPlan_Train2_1', 'FloorPlan_Train2_2', 'FloorPlan_Train2_3', 'FloorPlan_Train2_4', 'FloorPlan_Train2_5',
                'FloorPlan_Train3_1', 'FloorPlan_Train3_2', 'FloorPlan_Train3_3', 'FloorPlan_Train3_4', 'FloorPlan_Train3_5',
                'FloorPlan_Train4_1', 'FloorPlan_Train4_2', 'FloorPlan_Train4_3', 'FloorPlan_Train4_4', 'FloorPlan_Train4_5',
                'FloorPlan_Train5_1', 'FloorPlan_Train5_2', 'FloorPlan_Train5_3', 'FloorPlan_Train5_4', 'FloorPlan_Train5_5',
                'FloorPlan_Train6_1', 'FloorPlan_Train6_2', 'FloorPlan_Train6_3', 'FloorPlan_Train6_4', 'FloorPlan_Train6_5',
                'FloorPlan_Train7_1', 'FloorPlan_Train7_2', 'FloorPlan_Train7_3', 'FloorPlan_Train7_4', 'FloorPlan_Train7_5',
                'FloorPlan_Train8_1', 'FloorPlan_Train8_2', 'FloorPlan_Train8_3', 'FloorPlan_Train8_4', 'FloorPlan_Train8_5',
                'FloorPlan_Train9_1', 'FloorPlan_Train9_2', 'FloorPlan_Train9_3', 'FloorPlan_Train9_4', 'FloorPlan_Train9_5',
                'FloorPlan_Train10_1', 'FloorPlan_Train10_2', 'FloorPlan_Train10_3', 'FloorPlan_Train10_4', 'FloorPlan_Train10_5',
                'FloorPlan_Train11_1', 'FloorPlan_Train11_2', 'FloorPlan_Train11_3', 'FloorPlan_Train11_4', 'FloorPlan_Train11_5',
                'FloorPlan_Train12_1', 'FloorPlan_Train12_2', 'FloorPlan_Train12_3', 'FloorPlan_Train12_4', 'FloorPlan_Train12_5'
            ],
            'val': [
                'FloorPlan_Val1_1', 'FloorPlan_Val1_2', 'FloorPlan_Val1_3',
                'FloorPlan_Val2_1', 'FloorPlan_Val2_2', 'FloorPlan_Val2_3',
                'FloorPlan_Val3_1', 'FloorPlan_Val3_2', 'FloorPlan_Val3_3',
                'FloorPlan_Val4_1', 'FloorPlan_Val4_2', 'FloorPlan_Val4_3',
                'FloorPlan_Val5_1', 'FloorPlan_Val5_2', 'FloorPlan_Val5_3'
            ],
            'test': [
                'FloorPlan_Test1_1', 'FloorPlan_Test1_2',
                'FloorPlan_Test2_1', 'FloorPlan_Test2_2',
                'FloorPlan_Test3_1', 'FloorPlan_Test3_2',
                'FloorPlan_Test4_1', 'FloorPlan_Test4_2',
                'FloorPlan_Test5_1', 'FloorPlan_Test5_2',
                'FloorPlan_Test6_1', 'FloorPlan_Test6_2',
                'FloorPlan_Test7_1', 'FloorPlan_Test7_2'
            ]
        }
    
    def split_ithor_scenes(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[str]]:
        train_scenes = []
        val_scenes = []
        test_scenes = []
        
        for room_type, scenes in self.ithor_scenes.items():
            # Shuffle scenes for this room type
            shuffled_scenes = scenes.copy()
            random.shuffle(shuffled_scenes)
            
            n_total = len(shuffled_scenes)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_scenes.extend(shuffled_scenes[:n_train])
            val_scenes.extend(shuffled_scenes[n_train:n_train + n_val])
            test_scenes.extend(shuffled_scenes[n_train + n_val:])
            
            print(f"{room_type}: {n_train} train, {n_val} val, {n_total - n_train - n_val} test")
        
        random.shuffle(train_scenes)
        random.shuffle(val_scenes)
        random.shuffle(test_scenes)
        
        return {
            'train': train_scenes,
            'val': val_scenes,
            'test': test_scenes
        }
    
    def get_robothor_splits(self) -> Dict[str, List[str]]:
        return self.robothor_splits
    
    def save_splits(self, dataset: str, output_dir: str = './config/splits'):
        os.makedirs(output_dir, exist_ok=True)
        
        if dataset == 'ithor':
            splits = self.split_ithor_scenes()
        elif dataset == 'robothor':
            splits = self.get_robothor_splits()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        for split_name, scenes in splits.items():
            filename = os.path.join(output_dir, f'{dataset}_{split_name}_scenes.yaml')
            with open(filename, 'w') as f:
                yaml.dump({
                    'dataset': dataset,
                    'split': split_name,
                    'scenes': scenes,
                    'num_scenes': len(scenes)
                }, f, default_flow_style=False)
            print(f"Saved {split_name} split to {filename} ({len(scenes)} scenes)")
        
        # Save combined splits file
        combined_file = os.path.join(output_dir, f'{dataset}_splits.yaml')
        with open(combined_file, 'w') as f:
            yaml.dump(splits, f, default_flow_style=False)
        print(f"\nSaved combined splits to {combined_file}")
        
        return splits
    
    def verify_splits(self, splits: Dict[str, List[str]]):
        all_scenes = set()
        for split_name, scenes in splits.items():
            scene_set = set(scenes)
            
            if len(scenes) != len(scene_set):
                print(f"WARNING: Duplicate scenes in {split_name} split!")
            
            overlap = all_scenes.intersection(scene_set)
            if overlap:
                print(f"WARNING: {split_name} overlaps with other splits: {overlap}")
            
            all_scenes.update(scene_set)
        
        print(f"\nTotal unique scenes across all splits: {len(all_scenes)}")

if __name__ == "__main__":
    splitter = AI2THORDatasetSplitter(seed=42)
    
    print("=== iTHOR Dataset Splits ===")
    ithor_splits = splitter.save_splits('ithor')
    splitter.verify_splits(ithor_splits)
    
    print("\n=== RoboTHOR Dataset Splits ===")
    robothor_splits = splitter.save_splits('robothor')
    splitter.verify_splits(robothor_splits)