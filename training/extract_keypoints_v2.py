import cv2
import numpy as np
import os
from tqdm import tqdm
import imageio
import tensorflow as tf
import tensorflow_hub as hub

def load_movenet():
    """Load the MoveNet model from TensorFlow Hub."""
    model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/4')
    movenet = model.signatures['serving_default']
    return movenet

def process_image(movenet, image, image_size=(256, 256)):
    """Process a single image through MoveNet."""
    img = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), image_size[0], image_size[1])
    input_image = tf.cast(img, dtype=tf.int32)
    
    # Run model inference
    outputs = movenet(input_image)
    keypoints = outputs['output_0'].numpy()
    
    # Reshape keypoints and convert to desired format
    keypoints = keypoints[0, 0, :, :3]  # Take only x, y, confidence
    return keypoints.flatten()  # Flatten to [x1, y1, c1, x2, y2, c2, ...]

def extract_keypoints_from_gif(gif_path, output_dir, movenet):
    """Extract keypoints from a GIF file using MoveNet."""
    # Read GIF
    gif = imageio.mimread(gif_path)
    keypoints_sequence = []
    
    # Process each frame
    for frame in gif:
        # Convert to RGB if needed
        if frame.shape[-1] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # Get keypoints
        keypoints = process_image(movenet, frame)
        
        # Only add if we got valid keypoints
        if len(keypoints) == 51:  # 17 keypoints * 3 (x, y, confidence)
            keypoints_sequence.append(keypoints)
    
    # Convert to numpy array
    if keypoints_sequence:
        keypoints_array = np.array(keypoints_sequence)
        
        # Apply smoothing to reduce jitter
        smoothed_keypoints = np.zeros_like(keypoints_array)
        window_size = 3
        
        for i in range(len(keypoints_array)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(keypoints_array), i + window_size // 2 + 1)
            smoothed_keypoints[i] = np.mean(keypoints_array[start_idx:end_idx], axis=0)
        
        # Save keypoints
        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(gif_path))[0] + '_keypoints.npy')
        np.save(output_path, smoothed_keypoints)
        print(f"✅ Saved keypoints to {output_path}")
        return True
    
    return False

def main():
    print("\nLoading MoveNet model...")
    movenet = load_movenet()
    print("✅ Model loaded successfully!")
    
    # Create output directory if it doesn't exist
    output_dir = 'training/keypoints'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of GIF files
    gif_dir = 'training/gifs'
    gif_files = [f for f in os.listdir(gif_dir) if f.endswith('.gif')]
    
    print(f"\nProcessing {len(gif_files)} GIF files...")
    
    # Process each GIF file with progress bar
    success_count = 0
    for gif_file in tqdm(gif_files):
        gif_path = os.path.join(gif_dir, gif_file)
        if extract_keypoints_from_gif(gif_path, output_dir, movenet):
            success_count += 1
    
    print(f"\n✅ Successfully processed {success_count} out of {len(gif_files)} files")

if __name__ == "__main__":
    main() 