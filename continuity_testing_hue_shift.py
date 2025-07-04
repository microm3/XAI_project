import torch
import numpy as np
from data import get_dataset
from train_model_iso_img import load_model, evaluate, get_device
from XAI_project.XAI_hue_shift_analysis import MODEL_PATH, hue_shifts, create_hue_shifted_dataloader
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

def evaluate_with_noise(cnn, classifier, test_loader, device, relative_noise_std, data_std):
    cnn.eval()
    classifier.eval()
    
    all_preds = []
    all_labels = []
    
    absolute_noise_std = relative_noise_std * data_std
    
    with torch.no_grad():
        for images, stats, labels in test_loader:
            images, stats, labels = images.to(device), stats.to(device), labels.to(device)
            
            noise = torch.randn_like(images) * absolute_noise_std
            perturbed_images = images + noise
            perturbed_images = torch.clamp(perturbed_images, -3, 3)
            
            image_feats = cnn(perturbed_images)
            outputs = classifier(image_feats)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    return accuracy_score(all_labels, all_preds)

def find_noise_threshold(baseline_accuracy, cnn, classifier, hue_shifted_loader, device, data_std):
    current_noise = 0.01
    target_drop = baseline_accuracy * 0.05 # 5% drop in accuracy 
    
    while current_noise <= 3.0:
        perturbed_accuracy = evaluate_with_noise(
            cnn, classifier, hue_shifted_loader, device, current_noise, data_std
        )
        accuracy_drop = baseline_accuracy - perturbed_accuracy
        
        if accuracy_drop >= target_drop:
            return current_noise
            
        current_noise += 0.01
    
    return None

def test_hue_shift(shift_name, shift_degrees, cnn, classifier, device, data_std):
    hue_shifted_loader = create_hue_shifted_dataloader(shift_degrees)
    baseline_accuracy = evaluate(cnn, classifier, hue_shifted_loader, device)
    
    threshold = find_noise_threshold(
        baseline_accuracy, cnn, classifier, hue_shifted_loader, device, data_std
    )
    
    print(f"\n{shift_name} ({shift_degrees}°)")
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    if threshold is not None:
        print(f"Threshold for >5% accuracy drop: {threshold:.3f} (sd_noise/sd_data)")
    else:
        print("No threshold found")
    
    return threshold, baseline_accuracy

if __name__ == "__main__":
    device = get_device()
    cnn, classifier = load_model(path=MODEL_PATH, device=device)
    
    # to get data_std
    _, test_loader = get_dataset()
    
    all_images = []
    for images, _, _ in test_loader:
        all_images.append(images.numpy())
    all_images = np.vstack(all_images)
    data_std = np.std(all_images)
    
    results = []
    
    # test each hue shift
    for shift_name, shift_degrees in hue_shifts.items():
        threshold, baseline_acc = test_hue_shift(
            shift_name, shift_degrees, cnn, classifier, device, data_std
        )
        if threshold is not None:
            results.append(threshold)

    mean_threshold = np.mean(results)
    min_threshold = np.min(results)
    max_threshold = np.max(results)
    
    print(f"Mean: {mean_threshold:.3f}")
    print(f"Min: {min_threshold:.3f}")
    print(f"Max: {max_threshold:.3f}")
