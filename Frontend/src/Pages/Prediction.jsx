import React, { useState } from "react";
import { Brain, AlertCircle, CheckCircle, Loader } from "lucide-react";

const CLASSIFIERS = {
  'BP_Class': 'Blood Pressure',
  'Diabetes_Class': 'Diabetes',
  'Dyslipidemia_Class': 'Dyslipidemia (Cholesterol)'
};

const MODEL_TYPES = ['GradientBoosting', 'LogisticRegression', 'RandomForest'];

// API Base URL - change this if your backend is on a different port/host
const API_BASE_URL = 'http://localhost:3000';

// Field configurations - defines which fields should be dropdowns vs numeric inputs
const FIELD_CONFIG = {
  age: { type: 'number', label: 'Age', placeholder: 'Enter age (e.g., 45)' },
  sex: { 
    type: 'select', 
    label: 'Sex', 
    options: [
      { value: '1', label: 'Male' },
      { value: '0', label: 'Female' }
    ]
  },
  weight: { type: 'number', label: 'Weight (kg)', placeholder: 'Enter weight in kg' },
  height: { type: 'number', label: 'Height (cm)', placeholder: 'Enter height in cm' },
  BMI: { type: 'number', label: 'BMI', placeholder: 'Enter BMI' },
  smoking: { 
    type: 'select', 
    label: 'Smoking Status', 
    options: [
      { value: '0', label: 'Non-smoker' },
      { value: '1', label: 'Current smoker' },
      { value: '2', label: 'Former smoker' }
    ]
  },
  alcohol_consumption: { 
    type: 'select', 
    label: 'Alcohol Consumption', 
    options: [
      { value: '0', label: 'None' },
      { value: '1', label: 'Occasional' },
      { value: '2', label: 'Regular' },
      { value: '3', label: 'Heavy' }
    ]
  },
  physical_activity: { 
    type: 'select', 
    label: 'Physical Activity Level', 
    options: [
      { value: '0', label: 'Sedentary' },
      { value: '1', label: 'Light' },
      { value: '2', label: 'Moderate' },
      { value: '3', label: 'Active' },
      { value: '4', label: 'Very Active' }
    ]
  },
  family_history: { 
    type: 'select', 
    label: 'Family History', 
    options: [
      { value: '0', label: 'No' },
      { value: '1', label: 'Yes' }
    ]
  },
  cholesterol_medication: { 
    type: 'select', 
    label: 'Cholesterol Medication', 
    options: [
      { value: '0', label: 'No' },
      { value: '1', label: 'Yes' }
    ]
  }
};

function Prediction() {
  const [inputs, setInputs] = useState({
    age: "",
    sex: "",
    weight: "",
    height: "",
    BMI: "",
    smoking: "",
    alcohol_consumption: "",
    physical_activity: "",
    family_history: "",
    cholesterol_medication: "",
  });
  
  const [classifier, setClassifier] = useState('BP_Class');
  const [modelType, setModelType] = useState('GradientBoosting');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [predictionMode, setPredictionMode] = useState('single');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setInputs(prev => ({ ...prev, [name]: value }));
  };

  // Validate that all fields are filled
  const validateInputs = () => {
    const emptyFields = Object.entries(inputs).filter(([_, value]) => value === "");
    if (emptyFields.length > 0) {
      const fieldNames = emptyFields.map(([key]) => FIELD_CONFIG[key]?.label || key).join(", ");
      setError(`Please fill in all fields: ${fieldNames}`);
      return false;
    }
    return true;
  };

  // Convert string inputs to numbers
  const prepareInputData = () => {
    const numericInputs = {};
    for (const [key, value] of Object.entries(inputs)) {
      const numValue = parseFloat(value);
      if (isNaN(numValue)) {
        throw new Error(`Invalid numeric value for ${FIELD_CONFIG[key]?.label || key}: ${value}`);
      }
      numericInputs[key] = numValue;
    }
    return numericInputs;
  };

  const handleSinglePredict = async () => {
    if (!validateInputs()) return;

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const numericInputs = prepareInputData();
      console.log('Sending data:', numericInputs);

      const response = await fetch(
        `${API_BASE_URL}/api/predict/${classifier}?model=${modelType}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(numericInputs),
        }
      );
      
      const data = await response.json();
      console.log('Response:', data);
      
      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }
      
      setPrediction(data);
    } catch (err) {
      console.error('Error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handlePredictAll = async () => {
    if (!validateInputs()) return;

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const numericInputs = prepareInputData();
      console.log('Sending comprehensive prediction data:', numericInputs);

      // Get predictions for all classifiers with all models
      const allPredictions = {};
      
      for (const classifierKey of Object.keys(CLASSIFIERS)) {
        allPredictions[classifierKey] = {};
        
        for (const model of MODEL_TYPES) {
          try {
            const response = await fetch(
              `${API_BASE_URL}/api/predict/${classifierKey}?model=${model}`,
              {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(numericInputs),
              }
            );
            
            const data = await response.json();
            
            if (response.ok && data.success) {
              allPredictions[classifierKey][model] = {
                prediction: data.prediction,
                probabilities: data.probabilities,
                class_labels: data.class_labels
              };
            } else {
              allPredictions[classifierKey][model] = {
                error: data.error || 'Prediction failed'
              };
            }
          } catch (err) {
            allPredictions[classifierKey][model] = {
              error: err.message
            };
          }
        }
      }
      
      setPrediction({
        success: true,
        predictions: allPredictions,
        input: numericInputs
      });
      
    } catch (err) {
      console.error('Error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = () => {
    if (predictionMode === 'single') {
      handleSinglePredict();
    } else {
      handlePredictAll();
    }
  };

  const getResultClass = (pred) => {
    if (!pred) return '';
    const value = Array.isArray(pred) ? pred[0] : pred;
    return value === 1 ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200';
  };

  const getResultText = (pred) => {
    if (!pred) return 'N/A';
    const value = Array.isArray(pred) ? pred[0] : pred;
    return value === 1 ? 'Positive' : 'Negative';
  };

  const getResultIcon = (pred) => {
    if (!pred) return null;
    const value = Array.isArray(pred) ? pred[0] : pred;
    return value === 1 ? (
      <AlertCircle className="w-5 h-5 text-red-600" />
    ) : (
      <CheckCircle className="w-5 h-5 text-green-600" />
    );
  };

  const renderInputField = (key) => {
    const config = FIELD_CONFIG[key];
    
    if (config.type === 'select') {
      return (
        <div key={key} className="flex flex-col">
          <label className="font-semibold text-gray-700 mb-1">
            {config.label}
          </label>
          <select
            name={key}
            value={inputs[key]}
            onChange={handleChange}
            className="border border-gray-300 p-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-400 bg-white"
          >
            <option value="">Select {config.label}</option>
            {config.options.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      );
    }
    
    return (
      <div key={key} className="flex flex-col">
        <label className="font-semibold text-gray-700 mb-1">
          {config.label}
        </label>
        <input
          type="number"
          step="any"
          name={key}
          value={inputs[key]}
          onChange={handleChange}
          placeholder={config.placeholder}
          className="border border-gray-300 p-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-400"
        />
      </div>
    );
  };

  return (
    <section className="max-w-6xl mx-auto p-6">
      <div className="text-center mb-10">
        <div className="inline-flex p-4 bg-gradient-to-br from-purple-100 to-pink-100 rounded-full mb-4">
          <Brain className="w-12 h-12 text-purple-600" />
        </div>
        <h2 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
          AI Health Classification
        </h2>
        <p className="text-gray-600 text-lg">
          Enter patient data to receive AI-driven health risk predictions.
        </p>
      </div>

      <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
        {/* Prediction Mode Selection */}
        <div className="mb-6 flex gap-4">
          <button
            onClick={() => {
              setPredictionMode('single');
              setPrediction(null);
              setError(null);
            }}
            className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all ${
              predictionMode === 'single'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            Single Prediction
          </button>
          <button
            onClick={() => {
              setPredictionMode('all');
              setPrediction(null);
              setError(null);
            }}
            className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all ${
              predictionMode === 'all'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            Complete Analysis (All Models)
          </button>
        </div>

        {/* Single Classifier Options */}
        {predictionMode === 'single' && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
            <div>
              <label className="font-semibold text-gray-700 mb-2 block">
                Select Classifier:
              </label>
              <select
                value={classifier}
                onChange={(e) => setClassifier(e.target.value)}
                className="w-full border border-gray-300 p-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-400 bg-white"
              >
                {Object.entries(CLASSIFIERS).map(([key, label]) => (
                  <option key={key} value={key}>{label}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="font-semibold text-gray-700 mb-2 block">
                Select Model:
              </label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="w-full border border-gray-300 p-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-400 bg-white"
              >
                {MODEL_TYPES.map((model) => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            </div>
          </div>
        )}

        {/* All Models Info */}
        {predictionMode === 'all' && (
          <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-blue-800 text-sm">
              <strong>Complete Analysis Mode:</strong> This will test all {Object.keys(CLASSIFIERS).length} classifiers 
              with all {MODEL_TYPES.length} models ({Object.keys(CLASSIFIERS).length * MODEL_TYPES.length} total predictions).
            </p>
          </div>
        )}

        {/* Input Fields */}
        <div className="mb-6">
          <h3 className="font-semibold text-gray-800 mb-4 text-lg">Patient Information</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {Object.keys(inputs).map((key) => renderInputField(key))}
          </div>
        </div>

        {/* Submit Button */}
        <button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold px-8 py-4 rounded-xl transition-all duration-200 hover:scale-105 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <Loader className="w-5 h-5 animate-spin" />
              {predictionMode === 'all' ? 'Running Complete Analysis...' : 'Processing...'}
            </>
          ) : (
            <>
              {predictionMode === 'all' ? 'Run Complete Analysis' : `Predict ${CLASSIFIERS[classifier]}`}
            </>
          )}
        </button>

        {/* Error Display */}
        {error && (
          <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-600 mt-0.5" />
            <div>
              <h4 className="font-semibold text-red-800">Error</h4>
              <p className="text-red-700 text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Single Prediction Result */}
        {prediction && predictionMode === 'single' && prediction.success && (
          <div className="mt-6 space-y-4">
            <div className={`p-6 rounded-xl border-2 ${getResultClass(prediction.prediction)}`}>
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-bold text-gray-800 text-lg">
                  {CLASSIFIERS[prediction.classifier]} Result
                </h4>
                {getResultIcon(prediction.prediction)}
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-gray-700 font-semibold">Prediction:</span>
                  <span className="text-2xl font-bold">
                    {getResultText(prediction.prediction)}
                  </span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-gray-700 font-semibold">Model Used:</span>
                  <span className="text-gray-900">{prediction.model}</span>
                </div>

                {prediction.probabilities && (
                  <div className="mt-4 pt-4 border-t">
                    <span className="text-gray-700 font-semibold block mb-2">Confidence:</span>
                    <div className="grid grid-cols-2 gap-2">
                      {prediction.probabilities[0].map((prob, idx) => (
                        <div key={idx} className="bg-white p-3 rounded-lg">
                          <div className="text-xs text-gray-600">
                            {prediction.class_labels?.[idx] || `Class ${idx}`}
                          </div>
                          <div className="text-lg font-bold text-purple-600">
                            {(prob * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Complete Analysis Results - All Classifiers with All Models */}
        {prediction && predictionMode === 'all' && prediction.success && (
          <div className="mt-6 space-y-6">
            <h4 className="font-bold text-gray-800 text-xl mb-4">Complete Health Analysis Results</h4>
            
            {Object.entries(prediction.predictions).map(([classifierKey, models]) => (
              <div key={classifierKey} className="border-2 border-gray-200 rounded-xl p-6 bg-gray-50">
                <h5 className="font-bold text-gray-900 text-lg mb-4 flex items-center gap-2">
                  <Brain className="w-6 h-6 text-purple-600" />
                  {CLASSIFIERS[classifierKey]}
                </h5>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {Object.entries(models).map(([modelName, result]) => {
                    if (result.error) {
                      return (
                        <div key={modelName} className="p-4 rounded-lg border-2 bg-yellow-50 border-yellow-200">
                          <div className="flex items-center gap-2 mb-2">
                            <AlertCircle className="w-4 h-4 text-yellow-600" />
                            <span className="font-semibold text-gray-800">{modelName}</span>
                          </div>
                          <p className="text-xs text-yellow-700">Error: {result.error}</p>
                        </div>
                      );
                    }
                    
                    return (
                      <div
                        key={modelName}
                        className={`p-4 rounded-lg border-2 ${getResultClass(result.prediction)}`}
                      >
                        <div className="flex items-center justify-between mb-3">
                          <span className="font-semibold text-gray-800 text-sm">{modelName}</span>
                          {getResultIcon(result.prediction)}
                        </div>
                        
                        <div className="space-y-2">
                          <div>
                            <span className="text-xs text-gray-600">Result:</span>
                            <p className="font-bold text-lg">{getResultText(result.prediction)}</p>
                          </div>

                          {result.probabilities && (
                            <div className="pt-2 border-t space-y-1">
                              <span className="text-xs text-gray-600 block">Confidence:</span>
                              {result.probabilities[0].map((prob, idx) => (
                                <div key={idx} className="flex justify-between items-center">
                                  <span className="text-xs text-gray-600">
                                    {result.class_labels?.[idx] || `Class ${idx}`}:
                                  </span>
                                  <span className="font-bold text-purple-600 text-sm">
                                    {(prob * 100).toFixed(1)}%
                                  </span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

export default Prediction;