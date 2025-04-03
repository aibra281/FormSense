import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
import tensorflowjs as tfjs

# --- Load CSVs ---
csv_folder = "training/keypoints"  # <- FIXED path
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

X = []
y = []

for file in csv_files:
    df = pd.read_csv(os.path.join(csv_folder, file), header=None)
    X.append(df.values)
    label = os.path.splitext(file)[0].split('_keypoints')[0]  # e.g., air_bike
    y += [label] * len(df)

X = np.vstack(X)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1)

# --- Build Model ---
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# --- Save Model & Labels ---
model.save("training/exercise_classifier.h5")
np.save("training/exercise_labels.npy", label_encoder.classes_)

# --- Convert to TensorFlow.js format ---
tfjs.converters.save_keras_model(model, "public/exercise_model")

print("✅ Model and label encoder saved to public/exercise_model/")
