import torch
from torchvision import transforms
import numpy as np
import colorsys
from PIL import Image
from data import get_sample_by_idx, add_white_background, get_dataset
from train_model_iso_img import load_model, evaluate
import warnings
import os
import matplotlib.pyplot as plt
from data import create_or_load_dataframe, tab_preprocess
import pandas as pd
warnings.filterwarnings('ignore')


# MODEL_PATH = 'pokemon_model_images_old_without_seed.pt'
MODEL_PATH = 'pokemon_model_images.pt'

# NOTE not pretty, but hard coding the grayscale accuracy for now, too lazy to import 
GRAYSCALE_ACC =  0.2030

# from 0-360 degrees, 30 degree intervals
hue_shifts = {
    'original': 0,
    'hue_shift_30': 30, # e.g. red → orange
    'hue_shift_60': 60,
    'hue_shift_90': 90,
    'hue_shift_120': 120,
    'hue_shift_150': 150,
    'hue_shift_180': 180,    
    'hue_shift_210': 210,
    'hue_shift_240': 240,
    'hue_shift_270': 270,
    'hue_shift_300': 300,
    'hue_shift_330': 330,
    'hue_shift_360': 360 # just to verify that 0 and 360 returns same result. whiiich led to a lot of debugging. 
}

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def hue_shift_transform(hue_shift_degrees):
    
    # the below fanciness is because the original colour shift method didn´t return symmetric result :( And it took forever without the vector stuff 
    
    
    def adjust_hue(image):
        if torch.is_tensor(image):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
            image = torch.clamp(image, 0, 1)
            image = transforms.ToPILImage()(image)
        
        # Convert to numpy array and normalize to 0-1
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Vectorized RGB to HSV conversion
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        
        maxc = np.maximum(r, np.maximum(g, b))
        minc = np.minimum(r, np.minimum(g, b))
        
        # Value
        v = maxc
        
        # Saturation
        delta = maxc - minc
        s = np.where(maxc == 0, 0, delta / maxc)
        
        # Hue
        h = np.zeros_like(maxc)
        mask = delta != 0
        
        # Red is max
        mask_r = mask & (maxc == r)
        h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
        
        # Green is max
        mask_g = mask & (maxc == g)
        h[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
        
        # Blue is max
        mask_b = mask & (maxc == b)
        h[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
        
        h = h / 6.0  # Normalize to 0-1
        
        # Apply hue shift
        h_shifted = (h + (hue_shift_degrees / 360.0)) % 1.0
        
        # Vectorized HSV to RGB conversion
        h_i = (h_shifted * 6).astype(int)
        f = h_shifted * 6 - h_i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        # Initialize output arrays
        r_new = np.zeros_like(v)
        g_new = np.zeros_like(v)
        b_new = np.zeros_like(v)
        
        # Apply HSV to RGB conversion based on hue sector
        idx = h_i % 6
        r_new[idx == 0] = v[idx == 0]; g_new[idx == 0] = t[idx == 0]; b_new[idx == 0] = p[idx == 0]
        r_new[idx == 1] = q[idx == 1]; g_new[idx == 1] = v[idx == 1]; b_new[idx == 1] = p[idx == 1]
        r_new[idx == 2] = p[idx == 2]; g_new[idx == 2] = v[idx == 2]; b_new[idx == 2] = t[idx == 2]
        r_new[idx == 3] = p[idx == 3]; g_new[idx == 3] = q[idx == 3]; b_new[idx == 3] = v[idx == 3]
        r_new[idx == 4] = t[idx == 4]; g_new[idx == 4] = p[idx == 4]; b_new[idx == 4] = v[idx == 4]
        r_new[idx == 5] = v[idx == 5]; g_new[idx == 5] = p[idx == 5]; b_new[idx == 5] = q[idx == 5]
        
        # Combine channels and convert back to uint8
        shifted_img = np.stack([r_new, g_new, b_new], axis=2)
        shifted_img = np.clip(shifted_img * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(shifted_img)
    
    return transforms.Compose([
        transforms.Lambda(add_white_background),
        transforms.Resize((224, 224)),
        transforms.Lambda(adjust_hue),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_hue_shifted_dataloader(hue_shift_degrees):
    if hue_shift_degrees == 0:
        _, test_loader = get_dataset()
    else:
        hue_transform = hue_shift_transform(hue_shift_degrees)
        _, test_loader = get_dataset(image_transform=hue_transform)
    
    return test_loader


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
        
def analyze_hue_dependency():
    device = get_device()
    
    cnn, classifier = load_model(path=MODEL_PATH, device=device)
    print(f"loaded model {MODEL_PATH}")
    
    results = {}
    
    for shift_name, shift_degrees in hue_shifts.items():       
        print(f"\nTesting {shift_name}:")
        hue_shifted_loader = create_hue_shifted_dataloader(shift_degrees)
        accuracy = evaluate(cnn, classifier, hue_shifted_loader, device)
        
        results[shift_name] = {
            'accuracy': accuracy,
            'hue_shift': shift_degrees
        }
        
    generate_hue_analysis_report(results)
    
    return results


def generate_hue_analysis_report(results):
    baseline_accuracy = results['original']['accuracy']
    
    print(f"\nBASELINE (Original Images):")
    print(f"Accuracy: {baseline_accuracy:.4f}")
    
    print(f"\n{'Hue Shift':<15} {'Accuracy':<12} {'Drop %':<12}")
    print("-" * 45)
    
    for shift_name, data in results.items():
        if shift_name == 'original':
            continue
            
        accuracy_drop = baseline_accuracy - data['accuracy']
        accuracy_drop_pct = (accuracy_drop / baseline_accuracy) * 100
        
        print(f"{shift_name:<15} {data['accuracy']:<12.4f} {accuracy_drop_pct:<12.1f}")


def plot_hue_shift_results(results, plot_grayscal_acc=False):
    hue_degrees = []
    accuracies = []
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['hue_shift'])
    
    for shift_name, data in sorted_results:
        hue_degrees.append(data['hue_shift'])
        accuracies.append(data['accuracy'])
    
    plt.figure(figsize=(8, 4))
    plt.plot(hue_degrees, accuracies, 'bo-', linewidth=2, markersize=8)
    
    if plot_grayscal_acc:
        plt.axhline(y=GRAYSCALE_ACC, color='red', linestyle='--', linewidth=2)
    
    plt.xlabel('Hue Shift (degrees)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Model Performance vs Hue Shift\n(Pokemon Type Classification)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.xticks(hue_degrees, rotation=45)
    
    for i, (hue, acc) in enumerate(zip(hue_degrees, accuracies)):
        plt.annotate(f'{acc:.3f}', (hue, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    all_values = accuracies + ([GRAYSCALE_ACC] if plot_grayscal_acc else [])
    min_acc = min(all_values)
    max_acc = max(all_values)
    plt.ylim(min_acc - 0.05, max_acc + 0.05)
    
    plt.tight_layout()
    plt.savefig(f'hue_shift_analysis_grayscale_{plot_grayscal_acc}.png', dpi=300, bbox_inches='tight')

from data import deencode_types

def show_prediction_under_hue_shift(df, sample_idx=0, hue_shift_deg=120, device=None):
    """
    Show prediction for a single Pokémon before and after hue shift.
    Useful for checking if color changes prediction (e.g. Bulbasaur turns Fire).
    """
    if device is None:
        device = get_device()
    
    # Load model
    cnn, classifier = load_model(path=MODEL_PATH, device=device)
    cnn.eval()
    classifier.eval()

    # Load transforms
    original_transform = hue_shift_transform(0)
    shifted_transform = hue_shift_transform(hue_shift_deg)
    
    # Get images and labels
    original_img, original_stats, original_label = get_sample_by_idx(df, sample_idx, original_transform)
    shifted_img, shifted_stats, shifted_label = get_sample_by_idx(df, sample_idx, shifted_transform)

    # Prepare images for display
    def denorm(tensor_img):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        return torch.clamp(tensor_img * std + mean, 0, 1)

    def predict(img_tensor):
        img = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = classifier(cnn(img))
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float().squeeze(0).cpu()
        return [deencode_types()[i] for i in pred.nonzero(as_tuple=True)[0].tolist()]

    # Get predicted types
    results = []
    for shift_name, shift_deg in hue_shifts.items():
        transform = hue_shift_transform(shift_deg)
        img, stats, label = get_sample_by_idx(df, sample_idx, transform)
        pred_types = predict(img)
        results.append((shift_name, shift_deg, pred_types, img))

    #table
    print(f"\n{'Shift Name':<15} {'Deg':<5} Predicted Types")
    print("-" * 50)
    for name, deg, preds, _ in results:
        print(f"{name:<15} {deg:<5} {', '.join(preds) or '—'}")


    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()
    for ax, (name, deg, preds, img) in zip(axes, results):
        ax.imshow(denorm(img).permute(1,2,0).cpu())
        ax.axis('off')
        ax.set_title(f"{name}\n{deg}° → {', '.join(preds)}", fontsize=8)
      
    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("hueshiftall.png")



# same as in train_model_iso_img.evaluate, but modified for types 
def evaluate_per_type(cnn, classifier, test_loader, device):
    cnn.eval()
    classifier.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, stats, labels in test_loader:
            images, stats, labels = images.to(device), stats.to(device), labels.to(device)

            image_feats = cnn(images)
            combined = image_feats  # Same as in the original evaluate function
            outputs = classifier(combined)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # type specific 
    type_accuracies = []
    for type_idx in range(all_labels.shape[1]):
        type_preds = all_preds[:, type_idx]
        type_labels = all_labels[:, type_idx]
        
        # calculate accuracy for samples that have this type 
        type_mask = type_labels == 1
        if type_mask.sum() > 0:
            type_accuracy = (type_preds[type_mask] == type_labels[type_mask]).mean()
        else:
            type_accuracy = 0
        
        type_accuracies.append(type_accuracy)
    
    return type_accuracies

def analyze_type_color_dependency():
    device = get_device()
    cnn, classifier = load_model(path=MODEL_PATH, device=device)
    
    all_types = deencode_types()
    type_results = {type_name: [] for type_name in all_types}
    
    for shift_name, shift_degrees in hue_shifts.items():
        hue_shifted_loader = create_hue_shifted_dataloader(shift_degrees)
        
        type_accuracies = evaluate_per_type(cnn, classifier, hue_shifted_loader, device)
        
        for type_idx, type_name in enumerate(all_types):
            type_results[type_name].append({
                'hue_shift': shift_degrees,
                'accuracy': type_accuracies[type_idx],
                'shift_name': shift_name
            })
    
    color_dependency_results = []
    
    for type_name in all_types:
        type_data = type_results[type_name]
        original_acc = next(d['accuracy'] for d in type_data if d['hue_shift'] == 0)
        
        if original_acc > 0: 
            shifted_accs = [d['accuracy'] for d in type_data if d['hue_shift'] != 0]
            avg_shifted_acc = np.mean(shifted_accs)
            
            worst_shift_data = min(type_data, key=lambda x: x['accuracy'] if x['hue_shift'] != 0 else float('inf'))
            
            color_dependency_results.append({
                'type': type_name,
                'original_accuracy': original_acc,
                'avg_shifted_accuracy': avg_shifted_acc,
                'worst_hue_shift': worst_shift_data['hue_shift'],
                'worst_accuracy': worst_shift_data['accuracy'],
                'accuracy_drop_worst': original_acc - worst_shift_data['accuracy']
            })
            

    color_dependency_df = pd.DataFrame(color_dependency_results)
    color_dependency_df.to_csv('hue_shift_colour_dependency_per_type_results.csv', index=False)



if __name__ == "__main__":
    df = create_or_load_dataframe()
    df = tab_preprocess(df)
    # save_hue_shift_examples(df)

    #results = analyze_hue_dependency()
    # Blue → red
    # show_prediction_under_hue_shift(df, sample_idx=3, hue_shift_deg=120)
    #plot_hue_shift_results(results, plot_grayscal_acc=True)
    #plot_hue_shift_results(results, plot_grayscal_acc=False)
    
    analyze_type_color_dependency()


"""
new model with seed 

BASELINE (Original Images):
Accuracy: 0.5777

Hue Shift       Accuracy     Drop %      
---------------------------------------------
hue_shift_30    0.3944       31.7        
hue_shift_60    0.2042       64.7        
hue_shift_90    0.1265       78.1        
hue_shift_120   0.1230       78.7        
hue_shift_150   0.1311       77.3        
hue_shift_180   0.1381       76.1        
hue_shift_210   0.1299       77.5        
hue_shift_240   0.1253       78.3        
hue_shift_270   0.1485       74.3        
hue_shift_300   0.2599       55.0        
hue_shift_330   0.4432       23.3        
hue_shift_360   0.5800       -0.4     
"""




