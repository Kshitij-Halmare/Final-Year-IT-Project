const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://localhost:5000';

// Middleware
app.use(helmet());
app.use(cors({
  origin: '*', // In production, specify your frontend URL
  methods: ['GET', 'POST'],
  credentials: true
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Available classifiers
const AVAILABLE_CLASSIFIERS = {
  'BP_Class': ['GradientBoosting', 'LogisticRegression', 'RandomForest'],
  'Diabetes_Class': ['GradientBoosting', 'LogisticRegression', 'RandomForest'],
  'Dyslipidemia_Class': ['GradientBoosting', 'LogisticRegression', 'RandomForest']
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy',
    timestamp: new Date().toISOString(),
    service: 'Heart Health Classification API'
  });
});

// Get available classifiers
app.get('/api/classifiers', (req, res) => {
  res.json({
    success: true,
    classifiers: AVAILABLE_CLASSIFIERS,
    count: Object.keys(AVAILABLE_CLASSIFIERS).length
  });
});

// Single prediction endpoint
app.post('/api/predict/:classifier', async (req, res) => {
  try {
    const { classifier } = req.params;
    const { model } = req.query;
    const inputData = req.body;

    console.log('Received prediction request:', { classifier, model, inputData });

    // Validate classifier
    if (!AVAILABLE_CLASSIFIERS[classifier]) {
      return res.status(400).json({
        success: false,
        error: `Invalid classifier. Must be one of: ${Object.keys(AVAILABLE_CLASSIFIERS).join(', ')}`
      });
    }

    // Validate model type if provided
    if (model && !AVAILABLE_CLASSIFIERS[classifier].includes(model)) {
      return res.status(400).json({
        success: false,
        error: `Invalid model type. Must be one of: ${AVAILABLE_CLASSIFIERS[classifier].join(', ')}`
      });
    }

    // Validate input data
    if (!inputData || Object.keys(inputData).length === 0) {
      return res.status(400).json({
        success: false,
        error: 'No input data provided'
      });
    }

    // Build URL with optional model parameter
    let url = `${PYTHON_SERVICE_URL}/predict/${classifier}`;
    if (model) {
      url += `?model=${model}`;
    }

    console.log('Forwarding to Python service:', url);

    // Forward request to Python service
    const response = await axios.post(url, inputData, {
      headers: { 'Content-Type': 'application/json' },
      timeout: 10000
    });

    console.log('Python service response:', response.data);

    res.json({
      success: true,
      classifier,
      prediction: response.data.prediction,
      probabilities: response.data.probabilities,
      class_labels: response.data.class_labels,
      model: response.data.model,
      input: inputData,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Prediction error:', error.message);
    
    if (error.response) {
      console.error('Python service error:', error.response.data);
      return res.status(error.response.status).json({
        success: false,
        error: error.response.data.error || 'Prediction failed'
      });
    }
    
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        success: false,
        error: 'Python service is not running. Please start the Flask server.'
      });
    }
    
    res.status(500).json({
      success: false,
      error: 'Internal server error during prediction'
    });
  }
});

// Batch prediction endpoint
app.post('/api/predict-all', async (req, res) => {
  try {
    const inputData = req.body;

    console.log('Received batch prediction request:', inputData);

    if (!inputData || Object.keys(inputData).length === 0) {
      return res.status(400).json({
        success: false,
        error: 'No input data provided'
      });
    }

    // Forward request to Python service
    const response = await axios.post(
      `${PYTHON_SERVICE_URL}/predict-all`,
      inputData,
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 15000
      }
    );

    console.log('Python service batch response received');

    res.json({
      success: true,
      predictions: response.data.predictions,
      input: inputData,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Batch prediction error:', error.message);
    
    if (error.response) {
      return res.status(error.response.status).json({
        success: false,
        error: error.response.data.error || 'Batch prediction failed'
      });
    }
    
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        success: false,
        error: 'Python service is not running. Please start the Flask server.'
      });
    }
    
    res.status(500).json({
      success: false,
      error: 'Internal server error during batch prediction'
    });
  }
});

// Compare models endpoint
app.post('/api/compare-models/:classifier', async (req, res) => {
  try {
    const { classifier } = req.params;
    const inputData = req.body;

    if (!AVAILABLE_CLASSIFIERS[classifier]) {
      return res.status(400).json({
        success: false,
        error: `Invalid classifier. Must be one of: ${Object.keys(AVAILABLE_CLASSIFIERS).join(', ')}`
      });
    }

    if (!inputData || Object.keys(inputData).length === 0) {
      return res.status(400).json({
        success: false,
        error: 'No input data provided'
      });
    }

    const response = await axios.post(
      `${PYTHON_SERVICE_URL}/compare-models/${classifier}`,
      inputData,
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 15000
      }
    );

    res.json({
      success: true,
      classifier,
      models: response.data.models,
      input: inputData,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Model comparison error:', error.message);
    
    if (error.response) {
      return res.status(error.response.status).json({
        success: false,
        error: error.response.data.error || 'Model comparison failed'
      });
    }
    
    res.status(500).json({
      success: false,
      error: 'Internal server error during model comparison'
    });
  }
});

// Model info endpoint
app.get('/api/model-info/:classifier', async (req, res) => {
  try {
    const { classifier } = req.params;
    const { model } = req.query;

    if (!AVAILABLE_CLASSIFIERS[classifier]) {
      return res.status(400).json({
        success: false,
        error: `Invalid classifier. Must be one of: ${Object.keys(AVAILABLE_CLASSIFIERS).join(', ')}`
      });
    }

    let url = `${PYTHON_SERVICE_URL}/model-info/${classifier}`;
    if (model) {
      url += `?model=${model}`;
    }

    const response = await axios.get(url, { timeout: 5000 });

    res.json({
      success: true,
      ...response.data
    });

  } catch (error) {
    console.error('Model info error:', error.message);
    res.status(500).json({
      success: false,
      error: 'Failed to retrieve model information'
    });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    success: false,
    error: 'An unexpected error occurred'
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found'
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`\n🚀 Heart Health Classification API running on port ${PORT}`);
  console.log(`📍 Health check: http://localhost:${PORT}/health`);
  console.log(`🔗 Python service URL: ${PYTHON_SERVICE_URL}`);
  console.log(`\n📊 Available Classifiers:`);
  Object.keys(AVAILABLE_CLASSIFIERS).forEach(classifier => {
    console.log(`   - ${classifier}: ${AVAILABLE_CLASSIFIERS[classifier].join(', ')}`);
  });
  console.log('');
});