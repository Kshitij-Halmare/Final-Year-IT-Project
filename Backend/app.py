from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import numpy as np
import pandas as pd
from pathlib import Path
import traceback

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_DIR = os.getenv('MODEL_DIR', './models')

# Available classifiers and their models
AVAILABLE_CLASSIFIERS = {
    'BP_Class': ['GradientBoosting', 'LogisticRegression', 'RandomForest'],
    'Diabetes_Class': ['GradientBoosting', 'LogisticRegression', 'RandomForest'],
    'Dyslipidemia_Class': ['GradientBoosting', 'LogisticRegression', 'RandomForest']
}

# Default model selection for each classifier
DEFAULT_MODELS = {
    'BP_Class': 'GradientBoosting',
    'Diabetes_Class': 'GradientBoosting',
    'Dyslipidemia_Class': 'GradientBoosting'
}

# Cache for loaded models
model_cache = {}

def load_model(classifier, model_type):
    """Load model with caching and handle both dict and direct model formats"""
    cache_key = f"{classifier}__{model_type}"
    
    if cache_key not in model_cache:
        model_path = os.path.join(MODEL_DIR, f"classifier_{classifier}__{model_type}.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        loaded_obj = joblib.load(model_path)
        
        # Handle different model storage formats
        if isinstance(loaded_obj, dict):
            print(f"⚠️  Model loaded as dict. Keys: {loaded_obj.keys()}")
            
            # Try common dictionary keys
            if 'model' in loaded_obj:
                model = loaded_obj['model']
                print(f"✓ Extracted model from 'model' key")
            elif 'pipeline' in loaded_obj:
                model = loaded_obj['pipeline']
                print(f"✓ Extracted model from 'pipeline' key")
            elif 'classifier' in loaded_obj:
                model = loaded_obj['classifier']
                print(f"✓ Extracted model from 'classifier' key")
            elif 'estimator' in loaded_obj:
                model = loaded_obj['estimator']
                print(f"✓ Extracted model from 'estimator' key")
            else:
                # Print all keys to help debug
                print(f"❌ Could not find model in dict. Available keys: {list(loaded_obj.keys())}")
                raise ValueError(f"Model dict does not contain expected keys. Found: {list(loaded_obj.keys())}")
            
            model_cache[cache_key] = model
        else:
            # Direct model object
            model_cache[cache_key] = loaded_obj
            print(f"✓ Loaded model directly: {cache_key}")
    
    return model_cache[cache_key]

def prepare_input_dataframe(input_data):
    """Convert input dict/list to DataFrame with proper type handling"""
    if isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    else:
        df = pd.DataFrame([input_data])
    
    # Convert all columns to numeric, handling None/null values
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check for NaN values
    if df.isnull().any().any():
        nan_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Invalid or missing numeric values in fields: {', '.join(nan_cols)}")
    
    return df

def get_feature_names_from_model(model):
    """Extract feature names from model pipeline"""
    try:
        if hasattr(model, 'named_steps') and 'prep' in model.named_steps:
            prep = model.named_steps['prep']
            if hasattr(prep, 'transformers_'):
                feature_names = []
                for name, transformer, columns in prep.transformers_:
                    if columns != 'drop' and isinstance(columns, list):
                        feature_names.extend(columns)
                return feature_names
    except:
        pass
    
    # Try to get feature names from the model directly
    try:
        if hasattr(model, 'feature_names_in_'):
            return model.feature_names_in_.tolist()
    except:
        pass
    
    return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Heart Health Classification Service',
        'models_loaded': len(model_cache),
        'available_classifiers': list(AVAILABLE_CLASSIFIERS.keys())
    }), 200

@app.route('/classifiers', methods=['GET'])
def list_classifiers():
    """List all available classifiers"""
    return jsonify({
        'classifiers': AVAILABLE_CLASSIFIERS,
        'default_models': DEFAULT_MODELS
    }), 200

@app.route('/predict/<classifier>', methods=['POST'])
def predict_single(classifier):
    """Predict single classifier with default or specified model"""
    try:
        print(f"\n=== Prediction Request for {classifier} ===")
        
        # Validate classifier
        if classifier not in AVAILABLE_CLASSIFIERS:
            return jsonify({
                'error': f'Invalid classifier. Must be one of: {", ".join(AVAILABLE_CLASSIFIERS.keys())}'
            }), 400
        
        # Get input data
        input_data = request.get_json()
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        print(f"Input data received: {input_data}")
        
        # Get model type from query parameter or use default
        model_type = request.args.get('model', DEFAULT_MODELS[classifier])
        
        if model_type not in AVAILABLE_CLASSIFIERS[classifier]:
            return jsonify({
                'error': f'Invalid model type. Must be one of: {", ".join(AVAILABLE_CLASSIFIERS[classifier])}'
            }), 400
        
        print(f"Using model: {model_type}")
        
        # Load model
        model = load_model(classifier, model_type)
        print(f"Model type: {type(model)}")
        print(f"Model has predict: {hasattr(model, 'predict')}")
        
        # Prepare input
        X = prepare_input_dataframe(input_data)
        print(f"Prepared DataFrame shape: {X.shape}")
        print(f"DataFrame columns: {X.columns.tolist()}")
        
        # Make prediction
        predictions = model.predict(X)
        print(f"Predictions: {predictions}")
        
        # Get probability if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X)
                probabilities = proba.tolist()
                print(f"Probabilities: {probabilities}")
            except Exception as e:
                print(f"Could not get probabilities: {str(e)}")
        
        # Return results
        result = {
            'prediction': predictions.tolist(),
            'classifier': classifier,
            'model': model_type,
        }
        
        if probabilities:
            result['probabilities'] = probabilities
            result['class_labels'] = ['Negative', 'Positive']
        
        print(f"=== Request Successful ===\n")
        return jsonify(result), 200
        
    except ValueError as e:
        print(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except FileNotFoundError as e:
        print(f"Model not found: {str(e)}")
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict-all', methods=['POST'])
def predict_all():
    """Predict all classifiers at once using default models"""
    try:
        print("\n=== Batch Prediction Request ===")
        
        # Get input data
        input_data = request.get_json()
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        print(f"Input data: {input_data}")
        
        results = {}
        
        # Predict for each classifier
        for classifier in AVAILABLE_CLASSIFIERS.keys():
            try:
                model_type = DEFAULT_MODELS[classifier]
                print(f"\nProcessing {classifier} with {model_type}...")
                
                model = load_model(classifier, model_type)
                X = prepare_input_dataframe(input_data)
                predictions = model.predict(X)
                
                result = {
                    'prediction': predictions.tolist(),
                    'model': model_type
                }
                
                # Add probabilities if available
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(X)
                        result['probabilities'] = proba.tolist()
                        result['class_labels'] = ['Negative', 'Positive']
                    except Exception as e:
                        print(f"Could not get probabilities for {classifier}: {str(e)}")
                
                results[classifier] = result
                print(f"✓ {classifier} prediction: {predictions[0]}")
                
            except Exception as e:
                print(f"✗ {classifier} failed: {str(e)}")
                results[classifier] = {
                    'error': str(e)
                }
        
        print("=== Batch Request Complete ===\n")
        return jsonify({
            'predictions': results
        }), 200
        
    except Exception as e:
        print(f"Batch prediction failed: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/compare-models/<classifier>', methods=['POST'])
def compare_models(classifier):
    """Compare all models for a specific classifier"""
    try:
        if classifier not in AVAILABLE_CLASSIFIERS:
            return jsonify({
                'error': f'Invalid classifier. Must be one of: {", ".join(AVAILABLE_CLASSIFIERS.keys())}'
            }), 400
        
        input_data = request.get_json()
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        X = prepare_input_dataframe(input_data)
        results = {}
        
        # Test all models for this classifier
        for model_type in AVAILABLE_CLASSIFIERS[classifier]:
            try:
                model = load_model(classifier, model_type)
                predictions = model.predict(X)
                
                result = {
                    'prediction': predictions.tolist()
                }
                
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(X)
                        result['probabilities'] = proba.tolist()
                    except:
                        pass
                
                results[model_type] = result
                
            except Exception as e:
                results[model_type] = {'error': str(e)}
        
        return jsonify({
            'classifier': classifier,
            'models': results
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Model comparison failed: {str(e)}'}), 500

@app.route('/model-info/<classifier>', methods=['GET'])
def model_info(classifier):
    """Get model information"""
    try:
        if classifier not in AVAILABLE_CLASSIFIERS:
            return jsonify({
                'error': f'Invalid classifier. Must be one of: {", ".join(AVAILABLE_CLASSIFIERS.keys())}'
            }), 400
        
        model_type = request.args.get('model', DEFAULT_MODELS[classifier])
        
        if model_type not in AVAILABLE_CLASSIFIERS[classifier]:
            return jsonify({
                'error': f'Invalid model type. Must be one of: {", ".join(AVAILABLE_CLASSIFIERS[classifier])}'
            }), 400
        
        model = load_model(classifier, model_type)
        
        # Try to extract feature names
        feature_names = get_feature_names_from_model(model)
        
        info = {
            'classifier': classifier,
            'model_type': model_type,
            'available_models': AVAILABLE_CLASSIFIERS[classifier],
            'has_predict_proba': hasattr(model, 'predict_proba')
        }
        
        if feature_names:
            info['features'] = feature_names
            info['feature_count'] = len(feature_names)
        
        return jsonify(info), 200
        
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': f'Failed to load model info: {str(e)}'}), 500

@app.route('/inspect-model/<classifier>', methods=['GET'])
def inspect_model(classifier):
    """Debug endpoint to inspect model file structure"""
    try:
        if classifier not in AVAILABLE_CLASSIFIERS:
            return jsonify({
                'error': f'Invalid classifier. Must be one of: {", ".join(AVAILABLE_CLASSIFIERS.keys())}'
            }), 400
        
        model_type = request.args.get('model', DEFAULT_MODELS[classifier])
        model_path = os.path.join(MODEL_DIR, f"classifier_{classifier}__{model_type}.pkl")
        
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model file not found: {model_path}'}), 404
        
        loaded_obj = joblib.load(model_path)
        
        info = {
            'file_path': model_path,
            'object_type': str(type(loaded_obj)),
            'is_dict': isinstance(loaded_obj, dict)
        }
        
        if isinstance(loaded_obj, dict):
            info['dict_keys'] = list(loaded_obj.keys())
            info['dict_structure'] = {k: str(type(v)) for k, v in loaded_obj.items()}
        else:
            info['has_predict'] = hasattr(loaded_obj, 'predict')
            info['has_predict_proba'] = hasattr(loaded_obj, 'predict_proba')
        
        return jsonify(info), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Verify model directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"⚠️  Warning: Model directory not found: {MODEL_DIR}")
        print(f"Creating directory: {MODEL_DIR}")
        os.makedirs(MODEL_DIR, exist_ok=True)
    else:
        print(f"✅ Model directory: {MODEL_DIR}")
        # List available models
        pkl_files = list(Path(MODEL_DIR).glob('classifier_*.pkl'))
        print(f"✅ Found {len(pkl_files)} model files:")
        for pkl_file in pkl_files:
            print(f"   - {pkl_file.name}")
    
    # Run Flask app
    port = int(os.getenv('PORT', 5000))
    print(f"\n🚀 Python Classification Service starting on port {port}\n")
    app.run(host='0.0.0.0', port=port, debug=True)