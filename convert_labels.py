import numpy as np
import json
import os

# Make sure the output directory exists
os.makedirs('public/tfjs_model', exist_ok=True)

# Load the labels
print("Loading exercise_labels.npy...")
labels = np.load('exercise_labels.npy', allow_pickle=True)
print(f"Loaded {len(labels)} labels: {labels.tolist()}")

# Save to JSON
output_path = 'public/tfjs_model/exercise_labels.json'
with open(output_path, 'w') as f:
    json.dump(labels.tolist(), f)
print(f"Successfully saved labels to {output_path}") 