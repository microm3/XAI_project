import numpy as np
import shap
from train_model_iso_tab import load_model, get_dataset, get_device
from XAI_project.XAI_global_shap_analysis import create_shap_wrapper_model, get_feature_names
from util import type_names
import warnings

warnings.filterwarnings('ignore')
MODEL_PATH = 'pokemon_model_tabular.pt'

def get_feature_importance_ranking(shap_values):
    mean_abs_importance = np.mean(np.abs(shap_values), axis=0)
    ranking = np.argsort(mean_abs_importance)[::-1]
    return ranking

def top5_features_changed(original_ranking, perturbed_ranking):
    original_top5 = set(original_ranking[:5])
    perturbed_top5 = set(perturbed_ranking[:5])
    return original_top5 != perturbed_top5

def test_noise_level(original_data, original_ranking, explainer, data_std, noise_level, type_idx=None, seed=42):
    np.random.seed(seed)
    
    absolute_noise_std = noise_level * data_std
    noise = np.random.normal(0, absolute_noise_std, original_data.shape)
    perturbed_data = original_data + noise
    perturbed_shap_values = explainer(perturbed_data)
    
    if type_idx is None:
        perturbed_shap = perturbed_shap_values.values.mean(axis=2)
    else:
        perturbed_shap = perturbed_shap_values.values[:, :, type_idx]
    
    perturbed_ranking = get_feature_importance_ranking(perturbed_shap)
    
    return top5_features_changed(original_ranking, perturbed_ranking), perturbed_ranking

def find_noise_threshold(original_data, original_ranking, explainer, data_std, type_idx=None):
    # first use large steps, then finer, for efficiency
    coarse_step = 0.1
    current_noise = coarse_step
    
    while True:
        changed, new_ranking = test_noise_level(
            original_data, original_ranking, explainer, data_std, 
            current_noise, type_idx
        )
        
        if changed:
            coarse_lower = max(0.01, current_noise - coarse_step)
            coarse_upper = current_noise
            break
            
        current_noise += coarse_step
    
    fine_step = 0.01
    current_noise = coarse_lower
    
    while current_noise <= coarse_upper:
        changed, new_ranking = test_noise_level(
            original_data, original_ranking, explainer, data_std, 
            current_noise, type_idx
        )
        
        if changed:
            return current_noise, new_ranking
            
        current_noise += fine_step
    
    return coarse_upper, new_ranking

def test_shap_type(shap_values, type_name, feature_names, test_features, explainer, data_std, type_idx=None):
    if type_idx is None:
        type_shap = shap_values.values.mean(axis=2)  # global
    else:
        type_shap = shap_values.values[:, :, type_idx]  # per-type
    
    original_ranking = get_feature_importance_ranking(type_shap)
    original_top5_names = [feature_names[i] for i in original_ranking[:5]]
    
    print(f"\n{type_name}")
    print(f"Original top 5 features: {original_top5_names}")
    
    threshold, new_ranking = find_noise_threshold(
        test_features, original_ranking, explainer, data_std, type_idx
    )
    
    new_top5_names = [feature_names[i] for i in new_ranking[:5]]
    print(f"New top 5: {new_top5_names}")
    print(f"Threshold for explanation change: {threshold:.3f} (sd_noise/sd_data)")

if __name__ == "__main__":
    device = get_device()

    tab_net, classifier, device = load_model(path=MODEL_PATH, device=device) 
    _, test_loader = get_dataset()
    
    test_features = []
    test_labels = []
    
    for _, stats, labels in test_loader:
        test_features.append(stats.numpy())
        test_labels.append(labels.numpy())
    
    test_features = np.vstack(test_features)
    test_labels = np.vstack(test_labels)
    
    # to report noise_sd/data_sd
    data_std = np.std(test_features)
    feature_names = get_feature_names()
    
    model_wrapper = create_shap_wrapper_model(tab_net, classifier, device)
    explainer = shap.Explainer(model_wrapper, test_features) 
    shap_values = explainer(test_features)

    # test global
    test_shap_type(shap_values, "Global", feature_names, test_features, explainer, data_std)
    
    # test per-type
    for type_idx, type_name in enumerate(type_names):
        test_shap_type(shap_values, type_name, feature_names, test_features, explainer, data_std, type_idx)