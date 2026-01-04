// API Configuration
const API_URL = 'http://localhost:5002';

// Form submission handler
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Get form data
    const formData = {
        age: parseFloat(document.getElementById('age').value),
        bmi: parseFloat(document.getElementById('bmi').value),
        HbA1c_level: parseFloat(document.getElementById('HbA1c_level').value),
        blood_glucose_level: parseFloat(document.getElementById('blood_glucose_level').value),
        hypertension: parseInt(document.getElementById('hypertension').value),
        heart_disease: parseInt(document.getElementById('heart_disease').value),
        gender: parseInt(document.getElementById('gender').value),
        smoking_history: parseInt(document.getElementById('smoking_history').value)
    };

    // Show loading overlay
    showLoading();

    try {
        // Make API request
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
        // Display results
        displayResults(result);
    } catch (error) {
        console.error('Error:', error);
        alert('Error making prediction. Please ensure the API server is running on port 5002.');
    } finally {
        hideLoading();
    }
});

// Display prediction results
function displayResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    const resultCard = document.getElementById('resultCard');
    const resultIcon = document.getElementById('resultIcon');
    const resultLabel = document.getElementById('resultLabel');
    const resultConfidence = document.getElementById('resultConfidence');
    
    // Show results section with animation
    resultsSection.style.display = 'block';
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
    
    // Update result label and styling
    const isDiabetic = result.prediction === 1;
    resultLabel.textContent = result.prediction_label;
    resultIcon.textContent = isDiabetic ? '⚠️' : '✅';
    
    // Update confidence
    const confidencePercent = (result.confidence * 100).toFixed(1);
    resultConfidence.innerHTML = `<strong>${confidencePercent}%</strong>`;
    
    // Update result card styling
    if (isDiabetic) {
        resultCard.style.borderLeft = '5px solid var(--danger-color)';
        resultLabel.style.color = 'var(--danger-color)';
    } else {
        resultCard.style.borderLeft = '5px solid var(--secondary-color)';
        resultLabel.style.color = 'var(--secondary-color)';
    }
    
    // Update probability bars
    const nonDiabeticProb = (result.probability.non_diabetic * 100).toFixed(1);
    const diabeticProb = (result.probability.diabetic * 100).toFixed(1);
    
    document.getElementById('probNonDiabetic').textContent = `${nonDiabeticProb}%`;
    document.getElementById('probDiabetic').textContent = `${diabeticProb}%`;
    
    // Animate bars
    setTimeout(() => {
        document.getElementById('barNonDiabetic').style.width = `${nonDiabeticProb}%`;
        document.getElementById('barDiabetic').style.width = `${diabeticProb}%`;
    }, 100);
    
    // Display input summary
    displayInputSummary(result.input_features);
}

// Display input summary
function displayInputSummary(features) {
    const detailsGrid = document.getElementById('detailsGrid');
    detailsGrid.innerHTML = '';
    
    const featureLabels = {
        age: 'Age',
        bmi: 'BMI',
        HbA1c_level: 'HbA1c Level',
        blood_glucose_level: 'Blood Glucose',
        hypertension: 'Hypertension',
        heart_disease: 'Heart Disease',
        gender: 'Gender',
        smoking_history: 'Smoking History'
    };
    
    const featureFormatters = {
        hypertension: (val) => val === 1 ? 'Yes' : 'No',
        heart_disease: (val) => val === 1 ? 'Yes' : 'No',
        gender: (val) => val === 1 ? 'Male' : 'Female',
        smoking_history: (val) => {
            const labels = ['Never', 'Former', 'Current', 'Not Current', 'Ever'];
            return labels[val] || val;
        },
        bmi: (val) => val.toFixed(1),
        HbA1c_level: (val) => val.toFixed(1) + '%',
        blood_glucose_level: (val) => val + ' mg/dL',
        age: (val) => val + ' years'
    };
    
    for (const [key, value] of Object.entries(features)) {
        const detailItem = document.createElement('div');
        detailItem.className = 'detail-item';
        
        const label = document.createElement('div');
        label.className = 'detail-label';
        label.textContent = featureLabels[key] || key;
        
        const valueEl = document.createElement('div');
        valueEl.className = 'detail-value';
        valueEl.textContent = featureFormatters[key] ? featureFormatters[key](value) : value;
        
        detailItem.appendChild(label);
        detailItem.appendChild(valueEl);
        detailsGrid.appendChild(detailItem);
    }
}

// Reset form
function resetForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('resultsSection').style.display = 'none';
}

// Load sample data
function loadSample(type) {
    if (type === 'high-risk') {
        // High risk sample
        document.getElementById('age').value = 65;
        document.getElementById('bmi').value = 32.5;
        document.getElementById('HbA1c_level').value = 7.5;
        document.getElementById('blood_glucose_level').value = 180;
        document.getElementById('hypertension').value = 1;
        document.getElementById('heart_disease').value = 1;
        document.getElementById('gender').value = 1;
        document.getElementById('smoking_history').value = 2;
    } else if (type === 'low-risk') {
        // Low risk sample
        document.getElementById('age').value = 35;
        document.getElementById('bmi').value = 22.0;
        document.getElementById('HbA1c_level').value = 5.0;
        document.getElementById('blood_glucose_level').value = 90;
        document.getElementById('hypertension').value = 0;
        document.getElementById('heart_disease').value = 0;
        document.getElementById('gender').value = 0;
        document.getElementById('smoking_history').value = 0;
    }
}

// Show loading overlay
function showLoading() {
    document.getElementById('loadingOverlay').classList.add('active');
}

// Hide loading overlay
function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('active');
}

// Check API health on page load
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (!response.ok) {
            throw new Error('API not responding');
        }
        console.log('✅ API is running and healthy');
    } catch (error) {
        console.error('❌ API is not accessible:', error);
        const warningDiv = document.createElement('div');
        warningDiv.style.cssText = 'background: #fee2e2; border: 2px solid #ef4444; padding: 1rem; margin: 1rem 2rem; border-radius: 8px; color: #991b1b;';
        warningDiv.innerHTML = '<strong>⚠️ Warning:</strong> API server is not running. Please start it with: <code>python src/serve.py</code>';
        document.querySelector('.container').insertBefore(warningDiv, document.querySelector('.content-wrapper'));
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkApiHealth();
    console.log('🏥 Diabetes Prediction System loaded');
});
