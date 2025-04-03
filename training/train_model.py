import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflowjs as tfjs

def load_data():
    # Load keypoints and labels
    keypoints_dir = "training/keypoints"
    X = []
    y = []
    
    for file in os.listdir(keypoints_dir):
        if file.endswith("_keypoints.npy"):
            # Load keypoints
            keypoints = np.load(os.path.join(keypoints_dir, file))
            
            # Get label from filename
            label = file.replace("_keypoints.npy", "")
            
            # Add each frame as a separate sample
            for frame in keypoints:
                X.append(frame)
                y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save label encoder classes
    np.save("exercise_labels.npy", label_encoder.classes_)
    
    return X, y_encoded, label_encoder.classes_

def create_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Load and preprocess data
    print("Loading data...")
    X, y, labels = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train model
    print("\nCreating model...")
    model = create_model(input_shape=(X.shape[1],), num_classes=len(labels))
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            'exercise_classifier.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    # Convert to TensorFlow.js format
    print("\nConverting to TensorFlow.js format...")
    tfjs.converters.save_keras_model(
        model,
        'public/tfjs_model',
        weight_shard_size_bytes=1024*1024,
        quantization_dtype_map={'float16': '*'}
    )
    
    print("\n✅ Training complete! Model saved as 'exercise_classifier.h5'")
    print(f"✅ TensorFlow.js model saved in 'public/tfjs_model'")
    print(f"✅ Labels saved as 'exercise_labels.npy'")

if __name__ == "__main__":
    main()
