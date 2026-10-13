import numpy as np
import joblib
import sys
import os
import warnings
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# THIS PATCH IS TO RESTORE THE JOBLIB COMPATIBILITY WITH NUMPY 2.X TO NUMPY 1.X
sys.modules['numpy._core'] = np.core
sys.modules['numpy._core.multiarray'] = np.core.multiarray
sys.modules['numpy._core.numeric'] = np.core.numeric

# Filter by a specific substring in the warning message
warnings.filterwarnings("ignore", message=".*version.*|.*break.*", category=UserWarning)

class BlackBoxProbe:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.call_count = 0
        self.max_calls = 100  # THIS IS TO FIX THE CAP ON NUMBER OF TIMES TO CALL THE MODEL 

    def predict(self, X):
        if self.call_count >= self.max_calls:
            raise Exception(f"API limit exceeded: {self.call_count}/{self.max_calls}")
        self.call_count += 1
        return self.model.predict(X)

def recover_model(probe):
    # PUT [0,0,0] IN Y=A+B1*X1+B2*X2+B3*X3
    # SO WE CAN GET THE Y=A AT [0,0,0]
    X_zero = np.array([[0, 0, 0]])
    intercept = probe.predict(X_zero)[0]  # THIS [0] AT THE LAST GIVES NUMBER NOT A LIST AS OUTPUT
    
    # Task 2: Recover Coefficients
    # An identity matrix isolates each coefficient. 
    # predict([1,0,0]) = intercept + B1. Therefore, B1 = predict([1,0,0]) - intercept.
    X_identity = np.eye(3)
    raw_predictions = probe.predict(X_identity)
    coefficients = raw_predictions - intercept
    
    return intercept, coefficients

def estimate_model_error(probe):
    np.random.seed(42)
    X_random = np.random.rand(50, 3) * 100
    predictions = probe.predict(X_random)
    
    variance = np.var(predictions)
    return variance

def detect_non_linearity(probe, intercept):
    # Task 4: Detect Non-linearity
    # Testing the superposition principle: if linear, f(A) + f(B) - intercept = f(A + B)
    A = np.array([[1, 2, 3]])
    B = np.array([[4, 5, 6]])
    
    pred_A = probe.predict(A)[0]
    pred_B = probe.predict(B)[0]
    pred_AB = probe.predict(A + B)[0]
    
    expected = pred_A + pred_B - intercept
    is_linear = np.isclose(expected, pred_AB, rtol=1e-5)
    return not is_linear

def estimate_true_mse(probe, dataset_path):
    # Load the actual dataset downloaded from the provided links
    df = pd.read_csv(dataset_path)
    
    # Extract a subset of data to stay under the 100 API call limit
    subset = df.head(50)
    
    # Isolate the exact 3 input features the model expects
    X_real = subset.iloc[:, :3].values 
    
    # Isolate the ground-truth target variable (assuming it is the last column)
    y_real = subset.iloc[:, -1].values
    
    # Query the model with the real data
    y_pred = probe.predict(X_real)
    
    # Calculate the actual MSE
    mse = mean_squared_error(y_real, y_pred)
    return mse

def estimate_true_mse2(probe, dataset_path):
    # Load the whitespace-separated dataset (e.g., housing.csv)
    df = pd.read_csv(dataset_path, sep=r"\s+")
    
    # Extract a subset of data to stay under the 100 API call limit
    subset = df.head(50)
    
    # Isolate the exact 3 input features the model expects
    X_real = subset.iloc[:, :3].values 
    
    # Isolate the ground-truth target variable (assuming it is the last column)
    y_real = subset.iloc[:, -1].values
    
    # Query the model with the real data
    y_pred = probe.predict(X_real)
    
    # Calculate the actual MSE
    mse = mean_squared_error(y_real, y_pred)
    return mse

def plot_feature_responses(intercept, coefficients, model_name):
    # Generate linear feature values from -10 to 10
    x = np.linspace(-10, 10, 100)
    
    # Set up a 1x3 subplot grid matching the expected reference layout
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(f'Problem 1 - Linear Model Behavior ({model_name})', fontsize=14, y=1.05)
    
    for i in range(3):
        # Calculate the isolated linear response for feature i
        y = intercept + coefficients[i] * x
        
        # Format the subplot
        axes[i].plot(x, y, label='model output', color='blue', linewidth=2)
        axes[i].set_title(f'Feature x_{i+1} Response')
        axes[i].set_xlabel(f'x_{i+1}')
        axes[i].set_ylabel('Predicted y')
        axes[i].grid(True, linestyle='--', alpha=0.5)
        axes[i].legend(loc='upper left')
        
    save_filename = f'{model_name}_behavior.png'
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"Graph successfully saved as '{save_filename}' (absolute path: {os.path.abspath(save_filename)}).")
    plt.close()

if __name__ == "__main__":
    # ------------------------------------------------------------------------------------
    # Model 1: Concrete Strength
    probe = BlackBoxProbe("concrete_strength_20260901_164159_model.joblib")
    dataset_path = "Concrete_Data.csv"
    intercept, coefficients = recover_model(probe)
    variance = estimate_model_error(probe)
    is_non_linear = detect_non_linearity(probe, intercept)
    esti = estimate_true_mse(probe, dataset_path)
    
    print(f"Recovered Intercept: {intercept:.4f}")
    print(f"Recovered Coefficients: {coefficients}")
    print(f"Prediction Variance (50 samples): {variance:.2f}")
    if variance > 1000:
        print("Error Estimate: High variance detected (potential overfitting to scale).")
    else:
        print("Error Estimate: Low variance detected (potential high bias).")
    print(f"Non-linearity Detected: {is_non_linear}")
    print(f"API Calls Used: {probe.call_count}/100")
    print(f"MSE: {esti}")
    
    # Generate the 1x3 response plot for the Concrete model
    plot_feature_responses(intercept, coefficients, "concrete_strength")
    print("-" * 40)
    
    # -----------------------------------------------------------------------------------------
    # Model 2: Boston Housing
    probe2 = BlackBoxProbe("boston_housing_20260901_164311_model.joblib")
    dataset_path2 = "housing.csv"
    intercept2, coefficients2 = recover_model(probe2)
    variance2 = estimate_model_error(probe2)
    is_non_linear2 = detect_non_linearity(probe2, intercept2)
    esti2 = estimate_true_mse2(probe2, dataset_path2)
    
    print(f"Recovered Intercept: {intercept2:.4f}")
    print(f"Recovered Coefficients: {coefficients2}")
    print(f"Prediction Variance (50 samples): {variance2:.2f}")
    if variance2 > 1000:
        print("Error Estimate: High variance detected (potential overfitting to scale).")
    else:
        print("Error Estimate: Low variance detected (potential high bias).")
    print(f"Non-linearity Detected: {is_non_linear2}")
    print(f"API Calls Used: {probe2.call_count}/100")
    print(f"MSE: {esti2}")
    
    # Generate the 1x3 response plot for the Housing model
    plot_feature_responses(intercept2, coefficients2, "boston_housing")