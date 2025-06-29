import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from torchvision import transforms
from train_model_iso_img import load_model, get_device
from data import deencode_types
from XAI_project.colour_analysis.color_definitions import COLOR_DEFINITIONS
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = 'pokemon_model_images.pt'

def create_solid_color_image(color_rgb, size=(224, 224)):
    image = Image.new('RGB', size, color_rgb)
    return image

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def test_color_predictions():
    device = get_device()
    cnn, classifier = load_model(path=MODEL_PATH, device=device)
    cnn.eval()
    classifier.eval()
    
    all_types = deencode_types()
    
    results = []
    
    for color_name, rgb_value in COLOR_DEFINITIONS.items():
        # Create solid color image
        solid_image = create_solid_color_image(rgb_value)
        
        image_tensor = preprocess_image(solid_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = cnn(image_tensor)
            logits = classifier(image_features)
            probabilities = torch.sigmoid(logits)
            
        probs = probabilities.squeeze().cpu().numpy()
        
        results.append({
            'color': color_name,
            'rgb': rgb_value,
            'all_probs': probs
        })
    
    return results

def save_all_probabilities_csv(results, filename='predictions_per_colour.csv'):
    all_types = deencode_types()
    csv_data = []
    
    for result in results:
        color = result['color']
        probs = result['all_probs']
        
        row = {'color': color}
        
        for i, type_name in enumerate(all_types):
            row[f'{type_name}_probability'] = round(probs[i], 4)
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(filename, index=False)
    return df

if __name__ == "__main__":
    results = test_color_predictions()
    save_all_probabilities_csv(results) 