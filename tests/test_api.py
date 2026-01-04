"""
API Testing Script for Diabetes Prediction Service

Tests all API endpoints including health checks, predictions,
and batch predictions.

Usage:
    python tests/test_api.py
    
    Or with pytest:
    pytest tests/test_api.py -v
"""

import requests
import json
import sys

# API Configuration
API_BASE_URL = "http://localhost:5002"


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def print_success(message):
    """Print success message in green"""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_error(message):
    """Print error message in red"""
    print(f"{Colors.RED}✗ {message}{Colors.END}")


def print_info(message):
    """Print info message in blue"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.END}")


def test_health_check():
    """Test the /health endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Status Code: {response.status_code}")
            print_success(f"Response: {json.dumps(data, indent=2)}")
            
            # Validate response structure
            assert data['status'] == 'healthy', "Status should be 'healthy'"
            assert 'model_loaded' in data, "Response should contain 'model_loaded'"
            print_success("Health check passed!")
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Health check failed: {str(e)}")
        return False


def test_model_info():
    """Test the /info endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Model Info Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/info")
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Status Code: {response.status_code}")
            print_info("Model Information:")
            print(json.dumps(data, indent=2))
            print_success("Model info retrieved successfully!")
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Model info request failed: {str(e)}")
        return False


def test_single_prediction():
    """Test the /predict endpoint with a single instance"""
    print("\n" + "="*60)
    print("TEST 3: Single Prediction Endpoint")
    print("="*60)
    
    # Test case: Patient with diabetes risk factors
    test_data = {
        "age": 65,
        "bmi": 32.5,
        "HbA1c_level": 7.5,
        "blood_glucose_level": 180,
        "hypertension": 1,
        "heart_disease": 1,
        "gender": 1,
        "smoking_history": 2
    }
    
    print_info("Test Input:")
    print(json.dumps(test_data, indent=2))
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Status Code: {response.status_code}")
            print_success("Prediction Result:")
            print(json.dumps(data, indent=2))
            
            # Validate response structure
            assert 'prediction' in data, "Response should contain 'prediction'"
            assert 'prediction_label' in data, "Response should contain 'prediction_label'"
            assert 'probability' in data, "Response should contain 'probability'"
            assert 'confidence' in data, "Response should contain 'confidence'"
            
            print_success("Single prediction test passed!")
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Single prediction failed: {str(e)}")
        return False


def test_batch_prediction():
    """Test the /predict/batch endpoint"""
    print("\n" + "="*60)
    print("TEST 4: Batch Prediction Endpoint")
    print("="*60)
    
    # Test cases: Multiple patients
    test_data = {
        "instances": [
            {
                "age": 45,
                "bmi": 25.0,
                "HbA1c_level": 5.5,
                "blood_glucose_level": 100,
                "hypertension": 0,
                "heart_disease": 0,
                "gender": 0,
                "smoking_history": 0
            },
            {
                "age": 70,
                "bmi": 35.0,
                "HbA1c_level": 8.0,
                "blood_glucose_level": 200,
                "hypertension": 1,
                "heart_disease": 1,
                "gender": 1,
                "smoking_history": 3
            },
            {
                "age": 55,
                "bmi": 28.0,
                "HbA1c_level": 6.5,
                "blood_glucose_level": 140,
                "hypertension": 1,
                "heart_disease": 0,
                "gender": 0,
                "smoking_history": 1
            }
        ]
    }
    
    print_info(f"Testing with {len(test_data['instances'])} instances")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Status Code: {response.status_code}")
            print_success("Batch Prediction Results:")
            print(json.dumps(data, indent=2))
            
            # Validate response structure
            assert 'predictions' in data, "Response should contain 'predictions'"
            assert 'total_instances' in data, "Response should contain 'total_instances'"
            assert len(data['predictions']) == len(test_data['instances']), \
                "Number of predictions should match number of instances"
            
            print_success("Batch prediction test passed!")
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Batch prediction failed: {str(e)}")
        return False


def test_invalid_input():
    """Test API with invalid input"""
    print("\n" + "="*60)
    print("TEST 5: Invalid Input Handling")
    print("="*60)
    
    # Test case: Missing required features
    invalid_data = {
        "age": 45,
        "bmi": 25.0
        # Missing other required features
    }
    
    print_info("Testing with incomplete data (should fail):")
    print(json.dumps(invalid_data, indent=2))
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=invalid_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 400:
            data = response.json()
            print_success(f"Status Code: {response.status_code} (Expected)")
            print_success("Error Response:")
            print(json.dumps(data, indent=2))
            
            assert 'error' in data, "Error response should contain 'error' field"
            print_success("Invalid input handled correctly!")
            return True
        else:
            print_error(f"Expected 400, got {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Invalid input test failed: {str(e)}")
        return False


def test_nonexistent_endpoint():
    """Test request to non-existent endpoint"""
    print("\n" + "="*60)
    print("TEST 6: Non-existent Endpoint (404)")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/nonexistent")
        
        if response.status_code == 404:
            print_success(f"Status Code: {response.status_code} (Expected)")
            data = response.json()
            print_success("Error Response:")
            print(json.dumps(data, indent=2))
            print_success("404 handling works correctly!")
            return True
        else:
            print_error(f"Expected 404, got {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"404 test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all API tests"""
    print("\n" + "="*60)
    print("🧪 DIABETES PREDICTION API TESTING SUITE")
    print("="*60)
    print_info(f"Testing API at: {API_BASE_URL}")
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print_error("\n❌ Cannot connect to API server!")
        print_info("Please start the server first:")
        print_info("  python src/serve.py")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n❌ Error connecting to API: {str(e)}")
        sys.exit(1)
    
    # Run tests
    results = []
    results.append(("Health Check", test_health_check()))
    results.append(("Model Info", test_model_info()))
    results.append(("Single Prediction", test_single_prediction()))
    results.append(("Batch Prediction", test_batch_prediction()))
    results.append(("Invalid Input", test_invalid_input()))
    results.append(("404 Handling", test_nonexistent_endpoint()))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{test_name}: {status}{Colors.END}")
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print_success("\n🎉 All tests passed!")
        return 0
    else:
        print_error(f"\n⚠️  {total - passed} test(s) failed!")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
