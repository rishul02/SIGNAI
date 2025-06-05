import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.model_selection import train_test_split
import json

class ASLDatasetPreparer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
    def download_kaggle_dataset(self):
        """Instructions and code to download ASL dataset from Kaggle"""
        print("To download the ASL dataset from Kaggle:")
        print("1. Install Kaggle API: pip install kaggle")
        print("2. Set up Kaggle credentials (kaggle.json)")
        print("3. Run the following commands:")
        print()
        print("# ASL Alphabet Dataset")
        print("kaggle datasets download -d grassknoted/asl-alphabet")
        print()
        print("# Alternative: ASL Dataset")
        print("kaggle datasets download -d datamunge/sign-language-mnist")
        print()
        print("# Extract the downloaded files")
        print("unzip asl-alphabet.zip -d asl_dataset/")
        
        # Actual download code (uncomment when kaggle is set up)
        """
        import kaggle
        
        # Download ASL Alphabet dataset
        kaggle.api.dataset_download_files(
            'grassknoted/asl-alphabet',
            path='asl_dataset/',
            unzip=True
        )
        
        print("Dataset downloaded successfully!")
        """
    
    def process_kaggle_images(self, dataset_path):
        """Process images from Kaggle ASL dataset"""
        data = []
        
        train_path = os.path.join(dataset_path, 'asl_alphabet_train')
        
        if not os.path.exists(train_path):
            print(f"Dataset not found at {train_path}")
            print("Please download the dataset first using download_kaggle_dataset()")
            return None
        
        print("Processing ASL images...")
        
        for class_name in os.listdir(train_path):
            class_path = os.path.join(train_path, class_name)
            
            if os.path.isdir(class_path):
                print(f"Processing class: {class_name}")
                
                for i, img_file in enumerate(os.listdir(class_path)):
                    if i >= 500:  # Limit per class for faster processing
                        break
                        
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        
                        # Load image
                        image = cv2.imread(img_path)
                        if image is None:
                            continue
                            
                        # Convert to RGB
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Extract landmarks
                        landmarks = self.extract_hand_landmarks(image_rgb)
                        
                        if landmarks is not None:
                            # Resize image for CNN
                            image_resized = cv2.resize(image_rgb, (224, 224))
                            
                            data.append({
                                'image': image_resized,
                                'landmarks': landmarks,
                                'label': class_name,
                                'file_path': img_path
                            })
        
        print(f"Processed {len(data)} images successfully")
        return data
    
    def extract_hand_landmarks(self, image):
        """Extract hand landmarks using MediaPipe"""
        results = self.hands.process(image)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks)
        
        return None
    
    def create_training_dataset(self, data, output_dir='processed_data'):
        """Create training dataset from processed data"""
        os.makedirs(output_dir, exist_ok=True)
        
        images = []
        landmarks = []
        labels = []
        
        for item in data:
            images.append(item['image'])
            landmarks.append(item['landmarks'])
            labels.append(item['label'])
        
        # Convert to numpy arrays
        images = np.array(images)
        landmarks = np.array(landmarks)
        labels = np.array(labels)
        
        # Save processed data
        np.save(os.path.join(output_dir, 'images.npy'), images)
        np.save(os.path.join(output_dir, 'landmarks.npy'), landmarks)
        np.save(os.path.join(output_dir, 'labels.npy'), labels)
        
        # Save metadata
        metadata = {
            'num_samples': len(data),
            'num_classes': len(set(labels)),
            'classes': sorted(list(set(labels))),
            'image_shape': images[0].shape,
            'landmark_shape': landmarks[0].shape
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {output_dir}")
        print(f"Images shape: {images.shape}")
        print(f"Landmarks shape: {landmarks.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Classes: {metadata['classes']}")
        
        return images, landmarks, labels, metadata

def main():
    preparer = ASLDatasetPreparer()
    
    # Show download instructions
    preparer.download_kaggle_dataset()
    
    # Process dataset (uncomment when dataset is available)
    """
    dataset_path = 'asl_dataset'
    data = preparer.process_kaggle_images(dataset_path)
    
    if data:
        images, landmarks, labels, metadata = preparer.create_training_dataset(data)
        print("Dataset preparation completed!")
    """
    
    print("\nNext steps:")
    print("1. Download the ASL dataset using the instructions above")
    print("2. Run this script again to process the images")
    print("3. Use the processed data with train_asl_model.py")

if __name__ == "__main__":
    main()
