import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from collections import defaultdict
from data import create_or_load_dataframe, tab_preprocess, get_sample_by_idx, Pokemon, add_white_background
from train_model_iso_img import load_model
import warnings
import cv2
import numpy as np        
from util import type_names
import os
warnings.filterwarnings('ignore')

# from 0-360 degrees
hue_shifts = {
    'original': 0,
    'hue_shift_30': 30,      # e.g. red → orange
    'hue_shift_60': 60,      # red → yellow
    'hue_shift_120': 120,    # red → green
    'hue_shift_180': 180,    # red → cyan (complementary)
    'hue_shift_240': 240,    # red → blue
    'hue_shift_300': 300     # red → purple
}

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

# MODEL_PATH = 'pokemon_model_images_old_without_seed.pt'
MODEL_PATH = 'pokemon_model_images.pt'

def hue_shift_transform(hue_shift_degrees):
    def adjust_hue(image):
        if torch.is_tensor(image):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
            image = torch.clamp(image, 0, 1)
            image = transforms.ToPILImage()(image)
        
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # scale to use 360* hue
        hue_shift_cv = int(hue_shift_degrees * 179 / 360)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift_cv) % 180
        
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(rgb)
    
    return transforms.Compose([
        transforms.Lambda(add_white_background),
        transforms.Resize((224, 224)),
        transforms.Lambda(adjust_hue),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_hue_shifted_dataset(hue_shift_degrees, filter_grayscale=True):
    hue_transform = hue_shift_transform(hue_shift_degrees)
    
    df = create_or_load_dataframe()
    df = tab_preprocess(df)
    
    if filter_grayscale and hue_shift_degrees > 0:
        df = df[~df['image_path'].apply(is_grayscale_image)].reset_index(drop=True)
    
    unique_pokemon = df['pokemon_name'].unique()
    
    torch.manual_seed(42)
    pokemon_indices = torch.randperm(len(unique_pokemon))
    train_size = int(0.8 * len(unique_pokemon))
    
    train_pokemon = set(unique_pokemon[pokemon_indices[:train_size]])
    test_pokemon = set(unique_pokemon[pokemon_indices[train_size:]])
    
    test_data = []
    
    for idx in range(len(df)):
        pokemon_name = df.iloc[idx]['pokemon_name']
        if pokemon_name in test_pokemon:
            image, stats, label = get_sample_by_idx(df, idx, hue_transform)
            test_data.append((image, stats, label))
    
    test_dataset = Pokemon(test_data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    return test_loader, df

def is_grayscale_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img.convert('RGB'))
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    return np.array_equal(r, g) and np.array_equal(g, b)


def find_colorful_pokemon_sample(df):
    for idx in range(min(20, len(df))):
        try:
            pokemon_name = df.iloc[idx]['pokemon_name']
            image_path = df.iloc[idx]['image_path']
            
            if not is_grayscale_image(image_path):
                return idx, pokemon_name
        except:
            continue

def save_hue_shift_examples(df, output_dir="hue_shift_examples"):
    sample_idx, sample_pokemon_name = find_colorful_pokemon_sample(df)    
    
    for shift_name, shift_degrees in hue_shifts.items():
        transform = hue_shift_transform(shift_degrees)
        image, stats, label = get_sample_by_idx(df, sample_idx, transform)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_denorm = image * std + mean
        image_denorm = torch.clamp(image_denorm, 0, 1)
        
        image_pil = transforms.ToPILImage()(image_denorm)
        
        filename = f"{shift_name}_{sample_pokemon_name.replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        image_pil.save(filepath)
        
def evaluate_per_type(cnn, classifier, test_loader, device, df):
    cnn.eval()
    classifier.eval()
    
    type_correct = defaultdict(int)
    type_total = defaultdict(int)
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, stats, labels in test_loader:
            images = images.to(device)
            stats = stats.to(device)
            labels = labels.to(device)
            
            cnn_output = cnn(images)
            predictions = classifier(cnn_output)
            predicted_probs = torch.sigmoid(predictions)
            predicted_labels = (predicted_probs > 0.5).float()
            
            all_predictions.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Per-type accuracy
            for i in range(len(type_names)):
                type_mask = labels[:, i] == 1  # Pokemon that have this type
                if type_mask.sum() > 0:  # If any Pokemon in batch have this type
                    type_correct[type_names[i]] += (predicted_labels[type_mask, i] == labels[type_mask, i]).sum().item()
                    type_total[type_names[i]] += type_mask.sum().item()
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Per-label accuracy: each type prediction evaluated independently
    per_label_accuracy = np.mean(all_predictions == all_labels)
    
    # Exact match: only correct if ALL types are predicted correctly for each Pokemon
    exact_matches = np.all(all_predictions == all_labels, axis=1)
    exact_match_accuracy = np.mean(exact_matches)
    
    # For type-specific analysis, per-label makes more sense
    overall_accuracy = per_label_accuracy
    
    type_accuracies = {}
    for type_name in type_names:
        if type_total[type_name] > 0:
            type_accuracies[type_name] = type_correct[type_name] / type_total[type_name]
        else:
            type_accuracies[type_name] = 0.0
    
    return overall_accuracy, exact_match_accuracy, type_accuracies

def analyze_hue_dependency():
    device = get_device()
    
    df = create_or_load_dataframe()
    df = tab_preprocess(df)
    save_hue_shift_examples(df)
    
    cnn, classifier = load_model(path=MODEL_PATH, device=device)
    print(f"loaded model {MODEL_PATH}")
    
    results = {}
    
    for shift_name, shift_degrees in hue_shifts.items():
        test_loader, df = get_hue_shifted_dataset(shift_degrees)
        per_label_acc, exact_match_acc, type_accs = evaluate_per_type(cnn, classifier, test_loader, device, df)
        
        results[shift_name] = {
            'per_label_accuracy': per_label_acc,
            'exact_match_accuracy': exact_match_acc,
            'type_accuracies': type_accs,
            'hue_shift': shift_degrees
        }
        
    generate_hue_analysis_report(results)
    
    return results

def generate_hue_analysis_report(results):
    baseline_per_label = results['original']['per_label_accuracy']
    baseline_exact_match = results['original']['exact_match_accuracy']
    baseline_types = results['original']['type_accuracies']
    
    print(f"\nBASELINE (Original Images):")
    print(f"Per-label Accuracy: {baseline_per_label:.4f}")
    print(f"Exact Match Accuracy: {baseline_exact_match:.4f}")
    
    print(f"\n{'Hue Shift':<15} {'Per-Label Acc':<12} {'Exact Match':<12} {'Per-Label Drop %':<15}")
    print("-" * 70)
    
    for shift_name, data in results.items():
        if shift_name == 'original':
            continue
            
        per_label_drop = baseline_per_label - data['per_label_accuracy']
        per_label_drop_pct = (per_label_drop / baseline_per_label) * 100
        
        print(f"{shift_name:<15} {data['per_label_accuracy']:<12.4f} {data['exact_match_accuracy']:<12.4f} {per_label_drop_pct:<15.1f}%")
    
    type_vulnerabilities = defaultdict(list)
    
    for shift_name, data in results.items():
        if shift_name == 'original':
            continue
            
        for type_name, acc in data['type_accuracies'].items():
            if baseline_types[type_name] > 0:  # Only consider types with test samples
                drop = baseline_types[type_name] - acc
                drop_pct = (drop / baseline_types[type_name]) * 100 if baseline_types[type_name] > 0 else 0
                type_vulnerabilities[type_name].append((shift_name, drop_pct))
    
    # Sort types by average vulnerability
    avg_vulnerabilities = {}
    for type_name, drops in type_vulnerabilities.items():
        avg_drop = np.mean([drop for _, drop in drops])
        avg_vulnerabilities[type_name] = avg_drop
    
    sorted_types = sorted(avg_vulnerabilities.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nMOST HUE-SENSITIVE POKEMON TYPES:")
    print(f"{'Type':<12} {'Avg Drop %':<10} {'Most Vulnerable To'}")
    print("-" * 45)
    
    for type_name, avg_drop in sorted_types[:8]:  # Top 8 most vulnerable
        worst_shift = max(type_vulnerabilities[type_name], key=lambda x: x[1])
        print(f"{type_name:<12} {avg_drop:<10.1f} {worst_shift[0]} ({worst_shift[1]:.1f}%)")
    

if __name__ == "__main__":
    analyze_hue_dependency()
    