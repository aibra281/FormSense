import os
import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import pandas as pd

# Ensure the output directory exists
os.makedirs('public/models', exist_ok=True)

# Load the model first to check its expected input shape
model = tf.keras.models.load_model('training/form_model.h5')
expected_input_shape = model.layers[0].input_shape[1:]  # Get the expected input shape

# Get the input shape from the first CSV file in the keypoints directory
KEYPOINT_DIR = "training/keypoints"
first_file = next(f for f in os.listdir(KEYPOINT_DIR) if f.endswith(".csv"))
df = pd.read_csv(os.path.join(KEYPOINT_DIR, first_file), header=None)
actual_input_shape = (df.shape[1] - 1,)  # -1 because last column is the target

print(f"Model expects input shape: {expected_input_shape}")
print(f"Actual input shape from data: {actual_input_shape}")

# If shapes don't match, we need to adjust the input data
if expected_input_shape != actual_input_shape:
    print("Warning: Input shapes don't match. The model was trained with a different input shape.")
    print("Please ensure your input data matches the model's expected shape.")
    exit(1)

# Convert to TensorFlow.js format
tfjs.converters.save_keras_model(model, 'public/models')
