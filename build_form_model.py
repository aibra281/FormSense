import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
import numpy as np
import os

# Simulate dummy keypoint data (replace with real data later)
X = np.random.rand(100, 132)  # 33 keypoints × 4 features = 132
y = np.random.rand(100)

# Build the model
model = Sequential([
    Input(shape=(132,), name="input_layer"),  # ✅ baked-in input shape
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, name="output")
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5)

# Save as Keras H5 model
os.makedirs("models", exist_ok=True)
model.save("models/form_model.h5")

print("✅ Model saved to models/form_model.h5")
