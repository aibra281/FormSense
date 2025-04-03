import os
import zipfile
import numpy as np
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def extract_mmfit():
    """Extract the mm-fit dataset from zip file"""
    print("Extracting mm-fit dataset...")
    with zipfile.ZipFile('mm-fit.zip', 'r') as zip_ref:
        zip_ref.extractall('data/mm-fit')
    print("Extraction complete!")

def process_mmfit_data():
    """Process the mm-fit dataset and convert it to our format"""
    base_path = Path('data/mm-fit/mm-fit')
    processed_data = []
    
    # Process each workout directory
    for workout_dir in tqdm(list(base_path.glob('w*')), desc="Processing workouts"):
        try:
            # Load labels
            labels_file = workout_dir / f"{workout_dir.name}_labels.csv"
            if not labels_file.exists():
                continue
                
            # Read CSV with correct column names
            labels = pd.read_csv(labels_file, header=None, 
                               names=['start_frame', 'end_frame', 'reps', 'exercise'])
            
            # Load 3D poses
            poses_3d_file = workout_dir / f"{workout_dir.name}_pose_3d.npy"
            if not poses_3d_file.exists():
                continue
                
            poses_3d = np.load(poses_3d_file)
            # Reshape the poses to combine the first two dimensions (3 and num_frames)
            poses_3d = poses_3d.transpose(1, 0, 2)  # (num_frames, 3, 18)
            
            # Process each exercise in the workout
            for _, row in labels.iterrows():
                start_frame = int(row['start_frame'])
                end_frame = int(row['end_frame'])
                exercise = row['exercise']
                
                # Extract pose sequence for this exercise
                exercise_poses = poses_3d[start_frame:end_frame]
                
                # Convert poses to our format (17 keypoints)
                for pose in exercise_poses:
                    processed_pose = convert_pose_format(pose)
                    if processed_pose is not None:
                        processed_data.append({
                            'exercise': exercise,
                            'pose': processed_pose.tolist(),
                            'correct': True  # mm-fit contains correct form data
                        })
                        
        except Exception as e:
            print(f"Error processing {workout_dir}: {str(e)}")
            continue
    
    return processed_data

def convert_pose_format(pose):
    """Convert mm-fit pose format to our 17-keypoint format"""
    try:
        # mm-fit pose format is (3, 18) where:
        # First dimension (3) represents [frame_number, x, y]
        # Second dimension (18) represents different keypoints
        # We need to extract just the x, y coordinates and add a z coordinate
        
        # Extract x and y coordinates (ignore frame number)
        x = pose[1]  # x coordinates for all joints
        y = pose[2]  # y coordinates for all joints
        
        # Create z coordinates (set to 0 as they're not available)
        z = np.zeros_like(x)
        
        # Combine into 3D coordinates
        coords = np.stack([x, y, z], axis=1)  # Shape: (18, 3)
        
        # Create output array for 17 keypoints
        output_pose = np.zeros((17, 3))
        
        # Copy the first 17 keypoints (if available)
        num_keypoints = min(17, len(coords))
        output_pose[:num_keypoints] = coords[:num_keypoints]
        
        return output_pose
    except Exception as e:
        print(f"Error converting pose: {str(e)}")
        return None

def save_processed_data(data):
    """Save processed data to JSON file"""
    output_file = 'data/processed_mmfit.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data, f)
    print(f"Saved {len(data)} processed poses to {output_file}")

def main():
    # Create necessary directories
    os.makedirs('data/mm-fit', exist_ok=True)
    
    # Extract dataset if not already extracted
    if not os.path.exists('data/mm-fit'):
        extract_mmfit()
    
    # Process the data
    processed_data = process_mmfit_data()
    
    # Save processed data
    save_processed_data(processed_data)
    
    print(f"Processed {len(processed_data)} poses from mm-fit dataset")

if __name__ == '__main__':
    main() 