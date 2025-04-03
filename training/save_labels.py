import numpy as np
import json
import os

def save_labels():
    # Load labels from npy file
    labels = np.load('exercise_labels.npy', allow_pickle=True).tolist()
    
    # Create directory if it doesn't exist
    os.makedirs('public/tfjs_model', exist_ok=True)
    
    # Save as JSON
    with open('public/tfjs_model/exercise_labels.json', 'w') as f:
        json.dump(labels, f)
    
    print("✅ Labels saved as JSON!")

if __name__ == "__main__":
    save_labels() 