import tensorflow as tf
import tensorflowjs as tfjs
import os
import json
import numpy as np

class ModelConverter:
    def __init__(self):
        pass
    
    def convert_to_tensorflowjs(self, model_path, output_dir):
        """Convert Keras model to TensorFlow.js format"""
        print(f"Converting {model_path} to TensorFlow.js...")
        
        # Load the model
        model = tf.keras.models.load_model(model_path)
        
        # Convert to TensorFlow.js
        tfjs.converters.save_keras_model(model, output_dir)
        
        print(f"Model converted and saved to {output_dir}")
        
        # Create model info file
        model_info = {
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'num_parameters': model.count_params(),
            'model_type': 'classification'
        }
        
        with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
    
    def optimize_model(self, model_path, output_path):
        """Optimize model for web deployment"""
        print("Optimizing model for web deployment...")
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Convert to TensorFlow Lite for optimization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save optimized model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Optimized model saved to {output_path}")
        
        # Get model size
        original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        optimized_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        print(f"Original model size: {original_size:.2f} MB")
        print(f"Optimized model size: {optimized_size:.2f} MB")
        print(f"Size reduction: {((original_size - optimized_size) / original_size * 100):.1f}%")
    
    def create_web_ready_models(self, models_dir='models', output_dir='web_models'):
        """Convert all models to web-ready formats"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Model files to convert
        models_to_convert = [
            ('asl_cnn_model.h5', 'cnn'),
            ('asl_landmark_model.h5', 'landmark')
        ]
        
        for model_file, model_name in models_to_convert:
            model_path = os.path.join(models_dir, model_file)
            
            if os.path.exists(model_path):
                # Convert to TensorFlow.js
                tfjs_output = os.path.join(output_dir, f'{model_name}_tfjs')
                self.convert_to_tensorflowjs(model_path, tfjs_output)
                
                # Create optimized version
                tflite_output = os.path.join(output_dir, f'{model_name}_optimized.tflite')
                self.optimize_model(model_path, tflite_output)
            else:
                print(f"Model not found: {model_path}")
        
        # Create deployment configuration
        deployment_config = {
            'models': {
                'cnn': {
                    'path': 'cnn_tfjs/model.json',
                    'type': 'image_classification',
                    'input_shape': [224, 224, 3],
                    'preprocessing': 'normalize_0_1'
                },
                'landmark': {
                    'path': 'landmark_tfjs/model.json',
                    'type': 'landmark_classification',
                    'input_shape': [63],
                    'preprocessing': 'normalize_landmarks'
                }
            },
            'ensemble': {
                'enabled': True,
                'weights': {
                    'cnn': 0.6,
                    'landmark': 0.4
                }
            }
        }
        
        with open(os.path.join(output_dir, 'deployment_config.json'), 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        print("Web-ready models created successfully!")
        print(f"Models available in: {output_dir}")

def main():
    converter = ModelConverter()
    
    # Check if models exist
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print("Models directory not found!")
        print("Please train your models first using train_asl_model.py")
        return
    
    # Convert models
    converter.create_web_ready_models()
    
    print("\nDeployment instructions:")
    print("1. Copy the web_models directory to your Next.js public folder")
    print("2. Update your React component to load the TensorFlow.js models")
    print("3. Use the deployment_config.json for model configuration")

if __name__ == "__main__":
    main()
