import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints_from_gif(gif_path, output_dir):
    # Read GIF
    cap = cv2.VideoCapture(gif_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if not frames:
        print(f"Warning: No frames found in {gif_path}")
        return
    
    # Process each frame
    keypoints_list = []
    for frame in frames:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Extract keypoints (x, y, z)
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
            keypoints_list.append(keypoints)
    
    if keypoints_list:
        # Save keypoints
        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(gif_path))[0] + '_keypoints.npy')
        np.save(output_path, np.array(keypoints_list))
        print(f"Saved keypoints to {output_path}")

def main():
    # Create output directory
    output_dir = "training/keypoints"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all GIFs
    gif_dir = "training/gifs"
    gif_files = [f for f in os.listdir(gif_dir) if f.endswith('.gif')]
    
    print(f"Processing {len(gif_files)} GIFs...")
    for gif_file in tqdm(gif_files):
        gif_path = os.path.join(gif_dir, gif_file)
        extract_keypoints_from_gif(gif_path, output_dir)
    
    print("✅ Keypoint extraction complete!")

if __name__ == "__main__":
    main()
