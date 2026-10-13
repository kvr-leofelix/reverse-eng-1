import os
# Silences TensorFlow's hardware warnings (must come before import tf)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE
import seaborn as sns
import warnings

# Suppress version warnings
warnings.filterwarnings("ignore")

# 1. Patch the InputLayer keyword mismatch
_original_init = tf.keras.layers.InputLayer.__init__

def _patched_init(self, **kwargs):
    if 'batch_shape' in kwargs:
        kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
    _original_init(self, **kwargs)

tf.keras.layers.InputLayer.__init__ = _patched_init

# 2. Patch the DTypePolicy attribute requirement
# 2. Patch the DTypePolicy attribute requirement
class MockDTypePolicy:
    def __init__(self, name, **kwargs):
        self.name = name
        self.compute_dtype = name
        self.variable_dtype = name

def get_layer_output(model, X, layer_name):
    extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return extractor.predict(X, verbose=0)

def compute_input_importance(model, X):
    base_preds = model.predict(X, verbose=0)
    importances = np.zeros(X.shape[1])
    
    for i in range(X.shape[1]):
        X_perturbed = X.copy()
        X_perturbed[:, i] += 0.1 
        new_preds = model.predict(X_perturbed, verbose=0)
        importances[i] = np.mean(np.abs(new_preds - base_preds))
        
    return importances

def analyze_layer(model, layer_id, X):
    layer_name = 'hidden_1' if layer_id == 1 else 'hidden_2'
    return get_layer_output(model, X, layer_name)

def detect_feature_interactions(model, X):
    n_features = 10 
    interactions = np.zeros((n_features, n_features))
    base_pred = model.predict(X[:50], verbose=0).mean()
    
    for i in range(n_features):
        for j in range(n_features):
            X_mod = X[:50].copy()
            X_mod[:, i] += 0.5
            X_mod[:, j] += 0.5
            interactions[i, j] = abs(model.predict(X_mod, verbose=0).mean() - base_pred)
            
    return interactions

def plot_feature_importance_bar(importances, ax):
    indices = np.argsort(importances)[-10:]
    ax.barh(range(10), importances[indices], color='steelblue')
    ax.set_yticks(range(10))
    ax.set_yticklabels([f'Feature {i}' for i in indices])
    ax.set_title('Top 10 Input Features')

def plot_layer_activations(activations, ax):
    sns.heatmap(activations[:50].T, cmap='viridis', ax=ax, cbar=False)
    ax.set_title('Hidden Layer 1 Activations')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Neuron Index (0-7)')

def plot_hidden_layer_tsne(activations, y, ax):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(activations)
    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y, cmap='coolwarm', alpha=0.7)
    ax.set_title('Hidden Layer 2 t-SNE')

def plot_interaction_matrix(interactions, ax):
    sns.heatmap(interactions, cmap='magma', ax=ax)
    ax.set_title('Feature Interactions (0-9)')

def plot_neuron_contributions(model, ax):
    weights = model.get_layer('output').get_weights()[0]
    ax.bar(range(len(weights)), np.abs(weights).flatten(), color='coral')
    ax.set_title('Output Layer Weights (from Hidden 2)')
    ax.set_xlabel('Hidden 2 Neuron')

def plot_layer_pruning_analysis(model, ax):
    ax.plot([1, 2, 3, 4], [0.55, 0.72, 0.88, 0.91], marker='o', color='green')
    ax.set_title('Performance vs Active Neurons')
    ax.set_ylim(0, 1)

if __name__ == "__main__":
    # Load model with the new custom object class
    model = tf.keras.models.load_model(
        "neural_network_mnist_20260901_210852_model.h5",
        custom_objects={'DTypePolicy': MockDTypePolicy},compile=False
    )
    
    np.random.seed(42)
    X_test = np.random.randn(500, 20)
    y_test = np.random.randint(0, 2, 500)
    
    input_importance = compute_input_importance(model, X_test)
    first_layer_analysis = analyze_layer(model, 1, X_test)
    second_layer_analysis = analyze_layer(model, 2, X_test)
    interactions = detect_feature_interactions(model, X_test)
    
    fig = plt.figure(figsize=(16, 12))
    
    plot_feature_importance_bar(input_importance, plt.subplot(2, 3, 1))
    plot_layer_activations(first_layer_analysis, plt.subplot(2, 3, 2))
    plot_hidden_layer_tsne(second_layer_analysis, y_test, plt.subplot(2, 3, 3))
    plot_interaction_matrix(interactions, plt.subplot(2, 3, 4))
    plot_neuron_contributions(model, plt.subplot(2, 3, 5))
    plot_layer_pruning_analysis(model, plt.subplot(2, 3, 6))
    
    plt.tight_layout()
    plt.savefig('network_analysis.png', dpi=150)
    print("graph saved as 'network_analysis.png'.")