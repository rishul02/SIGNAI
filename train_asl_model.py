import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mediapipe as mp
import json

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class ASLDataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
        self.hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def load_kaggle_dataset(self):
        """Load ASL Alphabet dataset from Kaggle"""
        print("Loading ASL dataset...")
        
        # ASL Alphabet dataset structure
        train_dir = os.path.join(self.data_path, 'asl_alphabet_train')
        test_dir = os.path.join(self.data_path, 'asl_alphabet_test')
        
        images = []
        labels = []
        landmarks_data = []
        
        # Process training data
        if os.path.exists(train_dir):
            for class_name in os.listdir(train_dir):
                class_path = os.path.join(train_dir, class_name)
                if os.path.isdir(class_path):
                    print(f"Processing class: {class_name}")
                    
                    for img_file in os.listdir(class_path)[:500]:  # Limit for demo
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_path, img_file)
                            
                            # Load and preprocess image
                            image = cv2.imread(img_path)
                            if image is not None:
                                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                image_resized = cv2.resize(image_rgb, (224, 224))
                                
                                # Extract hand landmarks
                                landmarks = self.extract_landmarks(image_rgb)
                                
                                if landmarks is not None:
                                    images.append(image_resized)
                                    labels.append(class_name)
                                    landmarks_data.append(landmarks)
        
        print(f"Loaded {len(images)} images with {len(set(labels))} classes")
        return np.array(images), np.array(labels), np.array(landmarks_data)
    
    def extract_landmarks(self, image):
        """Extract hand landmarks using MediaPipe"""
        results = self.hands.process(image)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks)
        return None
    
    def preprocess_data(self, images, labels, landmarks):
        """Preprocess images and labels for training"""
        # Normalize images
        images = images.astype('float32') / 255.0
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        labels_categorical = to_categorical(labels_encoded)
        
        # Normalize landmarks
        landmarks = landmarks.astype('float32')
        
        return images, labels_categorical, landmarks, labels_encoded

class ASLModel:
    def __init__(self, num_classes, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.landmark_model = None
        
    def create_cnn_model(self):
        """Create CNN model for image-based ASL recognition"""
        model = tf.keras.Sequential([
            # Convolutional layers
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def create_landmark_model(self):
        """Create model for landmark-based ASL recognition"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(63,)),  # 21 landmarks * 3 coords
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.landmark_model = model
        return model
    
    def train_models(self, images, labels, landmarks, validation_split=0.2):
        """Train both CNN and landmark models"""
        # Split data
        X_img_train, X_img_val, X_land_train, X_land_val, y_train, y_val = train_test_split(
            images, landmarks, labels, test_size=validation_split, random_state=42
        )
        
        # Data augmentation for images
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False
        )
        
        # Train CNN model
        print("Training CNN model...")
        cnn_history = self.model.fit(
            datagen.flow(X_img_train, y_train, batch_size=32),
            epochs=20,
            validation_data=(X_img_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
            ]
        )
        
        # Train landmark model
        print("Training landmark model...")
        landmark_history = self.landmark_model.fit(
            X_land_train, y_train,
            epochs=50,
            batch_size=64,
            validation_data=(X_land_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        
        return cnn_history, landmark_history
    
    def save_models(self, model_dir='models'):
        """Save trained models"""
        os.makedirs(model_dir, exist_ok=True)
        
        if self.model:
            self.model.save(os.path.join(model_dir, 'asl_cnn_model.h5'))
            print("CNN model saved!")
        
        if self.landmark_model:
            self.landmark_model.save(os.path.join(model_dir, 'asl_landmark_model.h5'))
            print("Landmark model saved!")

def main():
    # Configuration
    DATA_PATH = "asl_dataset"  # Path to your downloaded Kaggle dataset
    
    # Initialize processor
    processor = ASLDataProcessor(DATA_PATH)
    
    # Create sample data for demonstration (replace with actual dataset loading)
    print("Creating sample ASL dataset...")
    
    # Sample ASL alphabet classes
    asl_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    # Generate sample data (in real scenario, use processor.load_kaggle_dataset())
    num_samples = 1000
    images = np.random.rand(num_samples, 224, 224, 3)
    labels = np.random.choice(asl_classes, num_samples)
    landmarks = np.random.rand(num_samples, 63)  # 21 landmarks * 3 coordinates
    
    print(f"Generated {len(images)} sample images")
    
    # Preprocess data
    images_processed, labels_processed, landmarks_processed, _ = processor.preprocess_data(
        images, labels, landmarks
    )
    
    # Create and train model
    num_classes = len(asl_classes)
    asl_model = ASLModel(num_classes)
    
    # Create models
    cnn_model = asl_model.create_cnn_model()
    landmark_model = asl_model.create_landmark_model()
    
    print("CNN Model Summary:")
    cnn_model.summary()
    
    print("\nLandmark Model Summary:")
    landmark_model.summary()
    
    # Train models
    print("Starting training...")
    cnn_history, landmark_history = asl_model.train_models(
        images_processed, labels_processed, landmarks_processed
    )
    
    # Save models
    asl_model.save_models()
    
    # Save label encoder
    import joblib
    joblib.dump(processor.label_encoder, 'models/label_encoder.pkl')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(cnn_history.history['accuracy'], label='CNN Train Accuracy')
    plt.plot(cnn_history.history['val_accuracy'], label='CNN Val Accuracy')
    plt.title('CNN Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(landmark_history.history['accuracy'], label='Landmark Train Accuracy')
    plt.plot(landmark_history.history['val_accuracy'], label='Landmark Val Accuracy')
    plt.title('Landmark Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    print("Training completed! Models saved in 'models' directory.")
    print("To use in your web app, convert models to TensorFlow.js format:")
    print("tensorflowjs_converter --input_format=keras models/asl_cnn_model.h5 web_models/cnn/")
    print("tensorflowjs_converter --input_format=keras models/asl_landmark_model.h5 web_models/landmark/")

if __name__ == "__main__":
    main()
