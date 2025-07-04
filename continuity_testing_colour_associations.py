import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from train_model_iso_img import load_model, get_device
from data import deencode_types
from XAI_global_colour_associations.color_definitions import COLOR_DEFINITIONS
import warnings

warnings.filterwarnings('ignore')
MODEL_PATH = 'pokemon_model_images.pt'

def create_solid_colour_image(colour_rgb, size=(224, 224)):
    image = Image.new('RGB', size, colour_rgb)
    return image

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def get_colour_predictions(cnn, classifier, colour_rgb, device):
    solid_image = create_solid_colour_image(colour_rgb)
    image_tensor = preprocess_image(solid_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = cnn(image_tensor)
        logits = classifier(image_features)
        probabilities = torch.sigmoid(logits)
    
    return probabilities.squeeze().cpu().numpy()

def get_top_predictions(probabilities, n=2):
    all_types = deencode_types()
    top_indices = np.argsort(probabilities)[::-1][:n]
    top_types = [all_types[i] for i in top_indices]
    top_probs = [probabilities[i] for i in top_indices]
    return top_types, top_probs

def perturb_colour(colour_rgb, noise_std):
    colour_array = np.array(colour_rgb, dtype=np.float32)
    noise = np.random.normal(0, noise_std, 3)
    perturbed = colour_array + noise
    perturbed = np.clip(perturbed, 0, 255)
    return tuple(perturbed.astype(int))

def top2_changed(original_top2, new_top2):
    return set(original_top2) != set(new_top2)

def find_threshold(cnn, classifier, colour_rgb, device):
    baseline_probs = get_colour_predictions(cnn, classifier, colour_rgb, device)
    baseline_top2, baseline_probs2 = get_top_predictions(baseline_probs, n=2)
    
    current_rgb_percentage = 0.1
    
    while current_rgb_percentage <= 5.0:
        absolute_rgb_noise = current_rgb_percentage * 255.0 / 100.0
        
        for i in range(10):
            np.random.seed(42 + int(current_rgb_percentage * 100) + i)
            perturbed_rgb = perturb_colour(colour_rgb, absolute_rgb_noise)
            perturbed_probs = get_colour_predictions(cnn, classifier, perturbed_rgb, device)
            perturbed_top2, _ = get_top_predictions(perturbed_probs, n=2)
            
            if top2_changed(baseline_top2, perturbed_top2):
                return current_rgb_percentage, baseline_top2, baseline_probs2
        
        current_rgb_percentage += 0.1
    
    return None, baseline_top2, baseline_probs2

def test_colour(colour_name, colour_rgb, cnn, classifier, device):
    threshold, baseline_top2, baseline_probs = find_threshold(
        cnn, classifier, colour_rgb, device
    )
    
    print(f"\n{colour_name} (RGB: {colour_rgb})")
    print(f"Top 2: {baseline_top2[0]} ({baseline_probs[0]:.3f}), {baseline_top2[1]} ({baseline_probs[1]:.3f})")
    if threshold is not None:
        print(f"Threshold: {threshold:.1f}% RGB noise")
    else:
        print("No threshold found")
    
    return threshold

if __name__ == "__main__":
    device = get_device()
    cnn, classifier = load_model(path=MODEL_PATH, device=device)
    cnn.eval()
    classifier.eval()
    
    results = []
    
    for colour_name, colour_rgb in COLOR_DEFINITIONS.items():
        threshold = test_colour(colour_name, colour_rgb, cnn, classifier, device)
        if threshold is not None:
            results.append(threshold)
    
    if results:
        mean_threshold = np.mean(results)
        min_threshold = np.min(results)
        max_threshold = np.max(results)
        
        print(f"\nSUMMARY:")
        print(f"Mean: {mean_threshold:.1f}% RGB")
        print(f"Range: {min_threshold:.1f}% - {max_threshold:.1f}% RGB")
        print(f"Valid results: {len(results)}/{len(COLOR_DEFINITIONS)}")
