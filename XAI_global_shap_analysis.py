import torch
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from train_model_iso_tab import load_model, get_dataset
from data import get_excluded_columns
import warnings
from util import type_names
import os
warnings.filterwarnings('ignore')

MODEL_PATH = 'pokemon_model_tabular.pt'
OUTPUT_DIR = 'shap_results'

def get_feature_names():
    df_sample = pd.read_csv('pokemon.csv')
    
    excluded_cols = get_excluded_columns()
    
    feature_names = [col for col in df_sample.columns if col not in excluded_cols]
    return feature_names

# wrapper function to return probabilities 
def create_shap_wrapper_model(tab_net, classifier, device):

    def model_wrapper(X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        with torch.no_grad():
            # Forward pass through tabular network and classifier
            feats = tab_net(X_tensor)
            logits = classifier(feats)
            probs = torch.sigmoid(logits)
        
        return probs.cpu().numpy()
    
    return model_wrapper

def analyze_pokemon_features_with_shap(output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    tab_net, classifier, device = load_model(path=MODEL_PATH)
    
    _, test_loader = get_dataset()
    
    test_features = []
    test_labels = []
    
    for _, stats, labels in test_loader:
        test_features.append(stats.numpy())
        test_labels.append(labels.numpy())
    
    test_features = np.vstack(test_features)
    test_labels = np.vstack(test_labels)
    
    feature_names = get_feature_names()
    print(f"Feature names ({len(feature_names)}): {feature_names}")
    
    model_wrapper = create_shap_wrapper_model(tab_net, classifier, device)
    
    explainer = shap.Explainer(model_wrapper, test_features) 
    
    shap_values = explainer(test_features)
    
    plt.figure(figsize=(12, 8))
    
    global_shap_values_signed = shap_values.values.mean(axis=2)
    
    shap.summary_plot(
        global_shap_values_signed, 
        test_features, 
        feature_names=feature_names,
        max_display=len(feature_names),
        show=False
    )
    plt.title("SHAP Feature Importance - Global (Signed Values)", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary_global_signed.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    for type_idx, type_name in enumerate(type_names):
        if type_name in type_names:
            plt.figure(figsize=(12, 8))
            
            shap.summary_plot(
                shap_values.values[:, :, type_idx], 
                test_features, 
                feature_names=feature_names,
                max_display=len(feature_names),
                show=False
            )
            plt.title(f"SHAP Feature Importance - {type_name} Type", fontsize=16, pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'shap_summary_{type_name.lower()}_type.png'), dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    analyze_pokemon_features_with_shap()