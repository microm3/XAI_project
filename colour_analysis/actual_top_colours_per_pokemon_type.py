import torch
import numpy as np
from PIL import Image
import cv2
from collections import defaultdict
from data import create_or_load_dataframe, tab_preprocess, deencode_types
import pandas as pd
import warnings
from XAI_project.colour_analysis.color_definitions import COLOR_NAMES, COLOR_RANGES

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='Number of distinct clusters')

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

# couldn't use mps gpu with sklearn.
def kmeans_torch(data, n_clusters, max_iters=50, device=None):
    if device is None:
        device = get_device()
    
    # Convert to PyTorch tensor and move to device
    X = torch.tensor(data, dtype=torch.float32, device=device)
    n_samples, n_features = X.shape
    
    # Initialize centroids randomly
    centroids = X[torch.randperm(n_samples, device=device)[:n_clusters]]
    
    for _ in range(max_iters):
        # Calculate distances to centroids
        distances = torch.cdist(X, centroids)
        
        # Assign each point to closest centroid
        labels = torch.argmin(distances, dim=1)
        
        # Update centroids
        new_centroids = torch.stack([X[labels == k].mean(0) if (labels == k).sum() > 0 
                                   else centroids[k] for k in range(n_clusters)])
        
        # Check for convergence
        if torch.allclose(centroids, new_centroids, atol=1e-4):
            break
            
        centroids = new_centroids
    
    return centroids.cpu().numpy(), labels.cpu().numpy()

def extract_dominant_colors_gpu(image_path, max_colors=3, sample_ratio=0.2, device=None):
    if device is None:
        device = get_device()
        
    # Load and preprocess image (keep on CPU for PIL operations)
    img = Image.open(image_path).convert('RGB')
    
    # Resize for faster processing
    if max(img.size) > 256:
        img.thumbnail((256, 256), Image.Resampling.LANCZOS)
    
    img_array = np.array(img, dtype=np.float32)
    
    # Remove white background efficiently
    mask = np.all(img_array > 240, axis=2)
    
    # Get non-white pixels
    pixels = img_array.reshape(-1, 3)
    non_white_pixels = pixels[~mask.flatten()]
    
    if len(non_white_pixels) < 20:
        return []
    
    # Sample pixels for faster processing
    if len(non_white_pixels) > 2000:
        sample_size = int(len(non_white_pixels) * sample_ratio)
        indices = np.random.choice(len(non_white_pixels), sample_size, replace=False)
        non_white_pixels = non_white_pixels[indices]
    
    n_clusters = min(max_colors, len(np.unique(non_white_pixels, axis=0)), 
                    len(non_white_pixels) // 15)
    
    if n_clusters < 1:
        return []
    
    centroids, labels = kmeans_torch(non_white_pixels, n_clusters, device=device)
    
    # Convert to HSV and create results
    dominant_colors_hsv = []
    
    for i, rgb in enumerate(centroids.astype(int)):
        # Ensure RGB values are in valid range
        rgb = np.clip(rgb, 0, 255)
        
        # Convert to HSV
        hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        h_degrees = (hsv[0] * 2) % 360
        s_percent = (hsv[1] / 255) * 100
        v_percent = (hsv[2] / 255) * 100
        
        # Count pixels in this cluster
        pixel_count = np.sum(labels == i)
        
        dominant_colors_hsv.append({
            'hue': h_degrees,
            'saturation': s_percent,
            'value': v_percent,
            'rgb': rgb.tolist(),
            'count': pixel_count
        })
    
    return dominant_colors_hsv

def batch_process_images(image_paths, pokemon_data, batch_size=32, device=None):
    if device is None:
        device = get_device()
    
    results = {}
    
    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        # Process each image in the batch
        for image_path in batch_paths:
            colors = extract_dominant_colors_gpu(image_path, device=device)
            
            if colors:
                pokemon_name = pokemon_data[image_path]['pokemon_name']
                pokemon_types = pokemon_data[image_path]['types']
                
                if pokemon_name not in results:
                    results[pokemon_name] = {
                        'colors': [],
                        'types': pokemon_types,
                        'image_count': 0
                    }
                
                results[pokemon_name]['colors'].extend(colors)
                results[pokemon_name]['image_count'] += 1
    
    return results

def analyze_pokemon_color_distributions_gpu(quick=True, sample_size=1000):
    device = get_device()
    
    df = create_or_load_dataframe()
    df = tab_preprocess(df)
    all_types = deencode_types()
    
    # Sample for quick analysis
    if quick and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # Prepare data for batch processing
    image_paths = []
    pokemon_data = {}
    
    for idx, row in df.iterrows():
        image_path = row['image_path']
        pokemon_name = row['pokemon_name']
        encoded_types = row['encoded_types']
        pokemon_types = [all_types[i] for i in range(len(all_types)) if encoded_types[i] == 1]
        
        image_paths.append(image_path)
        pokemon_data[image_path] = {
            'pokemon_name': pokemon_name,
            'types': pokemon_types
        }
    
    # Process images in batches
    pokemon_colors = batch_process_images(image_paths, pokemon_data, batch_size=16, device=device)
    
    # Create type-level summary
    type_colors = {type_name: [] for type_name in all_types}
    
    for pokemon_name, data in pokemon_colors.items():
        for ptype in data['types']:
            type_colors[ptype].extend(data['colors'])
    
    return type_colors, pokemon_colors

def extract_colors_simple(pixels, max_colors=3):
    # Use more aggressive binning for speed
    binned_pixels = (pixels // 64) * 64  # Reduce to 4 values per channel
    
    unique_colors, inverse_indices, counts = np.unique(
        binned_pixels.view(np.dtype((np.void, 3))), 
        return_inverse=True, 
        return_counts=True
    )
    
    # Get top colors
    top_indices = np.argsort(counts)[-max_colors:]
    dominant_colors = []
    
    for idx in top_indices:
        rgb = np.frombuffer(unique_colors[idx], dtype=np.uint8)
        
        # Convert to HSV
        hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        h_degrees = (hsv[0] * 2) % 360
        s_percent = (hsv[1] / 255) * 100
        v_percent = (hsv[2] / 255) * 100
        
        dominant_colors.append({
            'hue': h_degrees,
            'saturation': s_percent,
            'value': v_percent,
            'rgb': rgb.tolist(),
            'count': int(counts[idx])
        })
    
    return dominant_colors

def summarize_type_colors(type_colors):
    type_color_summary = {}
    
    # Use the shared color ranges
    color_ranges = COLOR_RANGES
    
    for type_name, colors in type_colors.items():
        if not colors:
            continue
            
        # Count colors in each range
        color_counts = defaultdict(int)
        total_colors = len(colors)
        
        for color in colors:
            hue = color['hue']
            saturation = color['saturation']
            value = color['value']
            
            # Handle special cases first
            if value < 20:  # Very dark
                color_counts['Black'] += 1
            elif saturation < 15:  # Very low saturation
                if value > 80:
                    color_counts['White'] += 1
                else:
                    color_counts['Gray'] += 1
            elif saturation < 40 and 15 <= hue <= 45 and 20 <= value <= 60:  # Brown
                color_counts['Brown'] += 1
            else:
                # Handle hue-based colors (only for saturated colors)
                if saturation > 30:
                    for color_name, hue_range in color_ranges.items():
                        if color_name in ['Brown', 'Gray', 'Black', 'White']:
                            continue
                            
                        if color_name == 'Red':  # Special case for red wrap-around
                            if any(min_h <= hue <= max_h for min_h, max_h in hue_range):
                                color_counts['Red'] += 1
                                break
                        else:
                            min_hue, max_hue = hue_range
                            if min_hue <= hue < max_hue:
                                color_counts[color_name] += 1
                                break
        
        # Calculate percentages
        color_percentages = {}
        total_counted = sum(color_counts.values())
        if total_counted > 0:
            for color_name, count in color_counts.items():
                color_percentages[color_name] = (count / total_counted) * 100
        
        type_color_summary[type_name] = {
            'total_images': total_colors,
            'color_percentages': color_percentages,
            'dominant_color': max(color_percentages.items(), key=lambda x: x[1]) if color_percentages else None
        }
    
    return type_color_summary

def export_type_color_probabilities(type_color_summary):
    color_names = COLOR_NAMES
    results = []
    
    for type_name, data in type_color_summary.items():
        if data['total_images'] > 10:  # Only include types with sufficient data
            row = {'type': type_name, 'total_images': data['total_images']}
            
            # Add percentage for each color
            for color_name in color_names:
                percentage = data['color_percentages'].get(color_name, 0)
                row[f'{color_name.lower()}_percentage'] = percentage
            
            results.append(row)
    
    return pd.DataFrame(results)

def main():
    type_colors, pokemon_colors = analyze_pokemon_color_distributions_gpu(quick=True, sample_size=1200)
    type_color_summary = summarize_type_colors(type_colors)
    probabilities_df = export_type_color_probabilities(type_color_summary)
    probabilities_df.to_csv('actual_top_colours_per_pokemon_type.csv', index=False)

if __name__ == "__main__":
    main() 