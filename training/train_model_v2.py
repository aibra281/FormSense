import numpy as np
import os
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflowjs as tfjs

def load_data(keypoints_dir='training/keypoints'):
    """Load keypoints and labels from the keypoints directory."""
    X = []
    y = []
    
    for file in os.listdir(keypoints_dir):
        if file.endswith('_keypoints.npy'):
            # Load keypoints
            keypoints = np.load(os.path.join(keypoints_dir, file))
            X.extend(keypoints)
            
            # Extract label from filename
            label = file.replace('_keypoints.npy', '')
            y.extend([label] * len(keypoints))
    
    # Convert to numpy arrays
    X = np.array(X)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Save labels for later use
    np.save('exercise_labels.npy', le.classes_)
    
    return X, y, le.classes_

def augment_data(X, y):
    """Apply data augmentation to the training data."""
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X)):
        # Original data
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        
        # Add random noise to keypoints
        noise = np.random.normal(0, 0.01, X[i].shape)
        augmented_X.append(X[i] + noise)
        augmented_y.append(y[i])
        
        # Mirror horizontally (flip left/right)
        mirrored = X[i].reshape(-1, 3)
        mirrored[:, 0] = 1 - mirrored[:, 0]  # Flip x coordinates
        augmented_X.append(mirrored.flatten())
        augmented_y.append(y[i])
        
        # Scale slightly
        scale_factor = np.random.uniform(0.95, 1.05)
        scaled = X[i].reshape(-1, 3)
        scaled[:, :2] *= scale_factor  # Scale x, y coordinates
        augmented_X.append(scaled.flatten())
        augmented_y.append(y[i])
        
        # Small random rotations
        for _ in range(2):
            angle = np.random.uniform(-0.1, 0.1)
            cos_t = np.cos(angle)
            sin_t = np.sin(angle)
            rotated = X[i].reshape(-1, 3).copy()
            x_coord = rotated[:, 0].copy()
            y_coord = rotated[:, 1].copy()
            rotated[:, 0] = x_coord * cos_t - y_coord * sin_t
            rotated[:, 1] = x_coord * sin_t + y_coord * cos_t
            augmented_X.append(rotated.flatten())
            augmented_y.append(y[i])
    
    return np.array(augmented_X), np.array(augmented_y)

def create_model(input_shape, num_classes):
    """Create an improved model architecture."""
    model = Sequential([
        Input(shape=input_shape),
        
        # First block - capture local features
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        
        # Second block - learn pose relationships
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dropout(0.3),
        
        # Third block - higher level features
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        
        # Fourth block - refined features
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        
        # Output layer
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def main():
    print("\nLoading data...")
    X, y, labels = load_data()
    
    print("\nAugmenting data...")
    X_aug, y_aug = augment_data(X, y)
    
    print("\nSplitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug
    )
    
    print("\nCreating model...")
    model = create_model(X_train.shape[1], len(np.unique(y)))
    
    # Compile model with learning rate schedule
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nSaving model...")
    model.save('exercise_classifier.h5')
    
    print("\nConverting to TensorFlow.js format...")
    tfjs.converters.save_keras_model(model, 'public/tfjs_model')
    
    # Save labels as JSON for frontend
    with open('public/tfjs_model/exercise_labels.json', 'w') as f:
        json.dump(labels.tolist(), f)
    
    print("\n✅ Training complete!")
    print("✅ Model saved as 'exercise_classifier.h5'")
    print("✅ TensorFlow.js model saved in 'public/tfjs_model'")
    print("✅ Labels saved in 'public/tfjs_model/exercise_labels.json'")

if __name__ == "__main__":
    main() 