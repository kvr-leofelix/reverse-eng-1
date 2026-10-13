import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import joblib
import warnings
from sklearn.linear_model import LogisticRegression

# Suppress version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Patch for NumPy 1.x vs 2.x compatibility
sys.modules['numpy._core'] = np.core
sys.modules['numpy._core.multiarray'] = np.core.multiarray
sys.modules['numpy._core.numeric'] = np.core.numeric

def recover_coefficients(model):
    X_test = np.array([[0, 0], [1, 0], [0, 1]])
    probs = model.predict_proba(X_test)[:, 1]
    log_odds = np.log(probs / (1 - probs))
    beta_0 = log_odds[0]
    beta_1 = log_odds[1] - beta_0
    beta_2 = log_odds[2] - beta_0
    return np.array([beta_0, beta_1, beta_2])

def compute_importance(model, base_pt=np.array([[0.5, 0.5]])):
    eps = 0.01
    base_p = model.predict_proba(base_pt)[0, 1]
    p_f1 = model.predict_proba(base_pt + np.array([[eps, 0]]))[0, 1]
    p_f2 = model.predict_proba(base_pt + np.array([[0, eps]]))[0, 1]
    return np.array([abs(p_f1 - base_p), abs(p_f2 - base_p)])

def plot_decision_boundary(model, ax, xx, yy, scaler):
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(scaler.transform(grid_points))[:, 1]
    Z_class = (probs >= 0.5).astype(int).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z_class, alpha=0.5, cmap='coolwarm')
    ax.set_title('Decision Boundary (Class 0 vs 1)')

def plot_feature_importance(model, ax):
    importance = compute_importance(model)
    ax.bar(['Feature 1', 'Feature 2'], importance, color=['blue', 'orange'])
    ax.set_title('Feature Importance (Perturbation Delta)')

def plot_confidence_heatmap(model, ax, xx, yy, scaler):
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(scaler.transform(grid_points))[:, 1]
    Z_prob = probs.reshape(xx.shape)
    
    contour = ax.contourf(xx, yy, Z_prob, levels=20, cmap='RdYlBu_r')
    ax.set_title('Model Confidence (Probability of Class 1)')
    return contour

def plot_coefficient_recovery(estimated_coef, true_coef, ax):
    x_labels = ['Intercept', 'Beta 1', 'Beta 2']
    x_pos = np.arange(len(x_labels))
    ax.bar(x_pos - 0.2, true_coef, 0.4, label='True (Scikit-Learn)')
    ax.bar(x_pos + 0.2, estimated_coef, 0.4, label='Recovered (Log-Odds)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.set_title('Coefficient Recovery Validation')

def detect_feature_interactions(model):
    # Test for non-linearity in the log-odds space
    # If standard logistic regression, log_odds(A) + log_odds(B) = log_odds(A+B)
    pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    probs = model.predict_proba(pts)[:, 1]
    
    # Avoid log(0) or log(1) warnings by clipping
    probs = np.clip(probs, 1e-10, 1 - 1e-10)
    lo = np.log(probs / (1 - probs))
    
    lo_base = lo[0]
    lo_A = lo[1] - lo_base
    lo_B = lo[2] - lo_base
    lo_AB = lo[3] - lo_base
    
    # If the combined effect differs from the sum of individual effects, interactions exist
    has_interaction = not np.isclose(lo_A + lo_B, lo_AB, rtol=1e-5)
    return has_interaction

def compute_boundary_sensitivity(model, scaler, grid_points):
    # Task 4: Find the point closest to the decision boundary (P = 0.5
    probs = model.predict_proba(scaler.transform(grid_points))[:, 1]
    boundary_idx = np.argmin(np.abs(probs - 0.5))
    boundary_pt = grid_points[boundary_idx].reshape(1, -1)
    
    # Apply a small adversarial perturbation
    epsilon = 0.05
    adversarial_pt = boundary_pt + np.array([[epsilon, -epsilon]])
    
    base_prob = model.predict_proba(scaler.transform(boundary_pt))[0, 1]
    adv_prob = model.predict_proba(scaler.transform(adversarial_pt))[0, 1]
    
    return abs(adv_prob - base_prob)



if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    X_real = df.iloc[:, 2:4].values
    y_real = df.iloc[:, 1].map({'M': 1, 'B': 0}).values

    payload = joblib.load("breast_cancer_20260901_204528_model.joblib")
    model = payload['model']
    scaler = payload['scaler']

    # Generate Meshgrid
    x_min, x_max = X_real[:, 0].min() - 1, X_real[:, 0].max() + 1
    y_min, y_max = X_real[:, 1].min() - 1, X_real[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Calculate baselines
    X_real_scaled = scaler.transform(X_real)
    true_coef = np.insert(model.coef_[0], 0, model.intercept_[0])
    estimated_coef = recover_coefficients(model)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    plot_decision_boundary(model, ax=axes[0, 0], xx=xx, yy=yy, scaler=scaler)
    plot_feature_importance(model, ax=axes[0, 1])
    contour = plot_confidence_heatmap(model, ax=axes[1, 0], xx=xx, yy=yy, scaler=scaler)
    fig.colorbar(contour, ax=axes[1, 0])
    plot_coefficient_recovery(estimated_coef, true_coef, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig('logistic_analysis.png', dpi=150)

    # Print Metrics
    coef_error = np.mean((estimated_coef - true_coef)**2)
    boundary_acc = np.mean(model.predict(X_real_scaled) == y_real)
    print(f"Coefficient Recovery Error: {coef_error:.4f}")
    print(f"Decision Boundary Accuracy: {boundary_acc:.4f}")
    # Calculate advanced metrics
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    interaction_detected = detect_feature_interactions(model)
    boundary_sensitivity = compute_boundary_sensitivity(model, scaler, grid_points)

    # Required Terminal Output
    print(f"Recovered Coefficients: {estimated_coef}")
    print(f"True Coefficients: {true_coef}")
    print(f"Coefficient Recovery Error: {coef_error:.4f}")
    print(f"Decision Boundary Accuracy: {boundary_acc:.4f}")
    print(f"Feature Interaction Detected: {interaction_detected}")
    print(f"Boundary Perturbation Sensitivity: {boundary_sensitivity:.4f}")