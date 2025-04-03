import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dataset:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.X = []
        self.y = []
        self.exercise_mapping = {}
        self.num_exercises = 0
        
    def load_data(self):
        """Load and combine data from both H36M and mm-fit datasets"""
        # Load mm-fit data
        self._load_mmfit_data()
        
        if len(self.X) == 0:
            raise ValueError("No data loaded from dataset")
            
        # Convert to numpy arrays
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
        # Normalize the data
        self._normalize_data()
        
        logger.info(f"Loaded {len(self.X)} total poses")
        logger.info(f"Found {self.num_exercises} unique exercises")
        
    def _load_mmfit_data(self):
        """Load data from processed mm-fit dataset"""
        mmfit_file = self.data_dir / 'processed_mmfit.json'
        if not mmfit_file.exists():
            logger.warning("Processed mm-fit data not found")
            return
            
        try:
            with open(mmfit_file, 'r') as f:
                data = json.load(f)
                
            # Create exercise mapping
            unique_exercises = sorted(list(set(item['exercise'] for item in data)))
            self.exercise_mapping = {ex: i for i, ex in enumerate(unique_exercises)}
            self.num_exercises = len(unique_exercises)
            
            logger.info(f"Exercise mapping: {self.exercise_mapping}")
            
            for item in data:
                pose = np.array(item['pose'])
                exercise_idx = self.exercise_mapping[item['exercise']]
                
                # Flatten the pose and append exercise index
                pose_flat = pose.reshape(-1)  # Shape: (51,) for 17 joints * 3 coordinates
                self.X.append(pose_flat)
                
                # Create one-hot encoded exercise label
                exercise_label = np.zeros(self.num_exercises)
                exercise_label[exercise_idx] = 1
                self.y.append(exercise_label)
                
            logger.info(f"Loaded {len(data)} poses from mm-fit dataset")
        except Exception as e:
            logger.error(f"Error loading mm-fit data: {str(e)}")
            
    def _normalize_data(self):
        """Normalize pose data to be scale and translation invariant"""
        # Center around hip joint (assuming it's the first joint)
        hip = self.X[:, :3]  # First joint (x,y,z)
        self.X = self.X.reshape(-1, 17, 3)  # Reshape to (N, 17, 3)
        self.X = self.X - hip[:, np.newaxis, :]  # Center around hip
        
        # Scale to unit size
        max_dist = np.max(np.sqrt(np.sum(self.X**2, axis=2)))
        self.X = self.X / max_dist
        
        # Flatten back to (N, 51)
        self.X = self.X.reshape(-1, 51)

def create_model(num_exercises):
    """Create a model for exercise classification"""
    # Input layer for pose
    pose_input = layers.Input(shape=(51,), name='pose_input')
    
    # Hidden layers
    x = layers.Dense(256, activation='relu')(pose_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer for exercise classification
    exercise_output = layers.Dense(num_exercises, activation='softmax', name='exercise_output')(x)
    
    # Create model
    model = models.Model(
        inputs=pose_input,
        outputs=exercise_output
    )
    
    # Compile model with classification loss and metrics
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Create dataset instance
    dataset = Dataset()
    
    try:
        # Load and preprocess data
        dataset.load_data()
        
        # Create and train model
        model = create_model(dataset.num_exercises)
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Train the model
        model.fit(
            dataset.X,
            dataset.y,  # Exercise labels
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    'models/exercise_classification_model.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        # Save exercise mapping
        with open('models/exercise_mapping.json', 'w') as f:
            json.dump(dataset.exercise_mapping, f, indent=2)
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 