import os
import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np

# Make sure the output directory exists
os.makedirs('public/tfjs_model', exist_ok=True)

# Load the model
print("Loading exercise_classifier.h5...")
model = tf.keras.models.load_model('exercise_classifier.h5')

# Print model summary to verify
print("\nModel Summary:")
model.summary()

# Convert to TensorFlow.js format with optimization
print("\nConverting to TensorFlow.js format...")
tfjs.converters.save_keras_model(
    model,
    'public/tfjs_model',
    weight_shard_size_bytes=1024*1024,  # 1MB shard size
    quantization_dtype_map={'float16': '*'}  # Use float16 for better performance
)

print("\n✅ Model converted successfully!") 