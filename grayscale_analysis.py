import torch
from torchvision import transforms
from data import get_dataset, create_or_load_dataframe, tab_preprocess, get_sample_by_idx, Pokemon, add_white_background
from torch.utils.data import random_split
from train_model_iso_img import load_model, evaluate
import warnings
warnings.filterwarnings('ignore')

# MODEL_PATH = 'pokemon_model_images_old_without_seed.pt'
MODEL_PATH = 'pokemon_model_images.pt'

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device

def get_grayscale_dataset():
    grayscale_transform = transforms.Compose([
        transforms.Lambda(add_white_background),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    df = create_or_load_dataframe()
    df = tab_preprocess(df)

    data = []
    for idx in range(len(df)):
        image, stats, label = get_sample_by_idx(df, idx, grayscale_transform)
        data.append((image, stats, label))

    dataset = Pokemon(data)
    train_size = int(0.8 * len(dataset))
    train_ds, test_ds = random_split(dataset, [train_size, len(dataset) - train_size])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)
    
    return test_loader

def analyze_color_dependency():
    device = get_device()
    
    cnn, classifier = load_model(path=MODEL_PATH, device=device)
    print(f"model loaded from {MODEL_PATH}")
    
    _, test_loader_rgb = get_dataset()
    print("Testing on RGB images:")
    rgb_accuracy = evaluate(cnn, classifier, test_loader_rgb, device)
    
    test_loader_gray = get_grayscale_dataset()
    print("\nTesting on grayscale images:")
    gray_accuracy = evaluate(cnn, classifier, test_loader_gray, device)
    
    color_dependency = rgb_accuracy - gray_accuracy
    color_dependency_pct = (color_dependency / rgb_accuracy) * 100
    
    print(f"RGB Accuracy:      {rgb_accuracy:.4f}")
    print(f"Grayscale Accuracy: {gray_accuracy:.4f}")
    print(f"Color Dependency:   {color_dependency:.4f} ({color_dependency_pct:.1f}%)")

if __name__ == "__main__":
    analyze_color_dependency()
    
    
    
"""
old data code, without new split code and seed
Testing on RGB images:
Per-label accuracy: 0.9927
Accuracy:  0.9026
Precision: 0.9721
Recall:    0.9401
F1-score:  0.9554
AUC-ROC:   0.9688

Testing on grayscale images:
Per-label accuracy: 0.9519
Accuracy:  0.4594
Precision: 0.8496
Recall:    0.6106
F1-score:  0.6682
AUC-ROC:   0.7971

RGB Accuracy:      0.9026
Grayscale Accuracy: 0.4594
Color Dependency:   0.4432 (49.1%)
"""


""" 
new with seed 

model loaded from pokemon_model_images.pt
Testing on RGB images:
Per-label accuracy: 0.9640
Accuracy:  0.5777
Precision: 0.8379
Recall:    0.6895
F1-score:  0.7504
AUC-ROC:   0.8385

Testing on grayscale images:
Per-label accuracy: 0.9512
Accuracy:  0.4292
Precision: 0.8144
Recall:    0.5974
F1-score:  0.6603
AUC-ROC:   0.7913
RGB Accuracy:      0.5777
Grayscale Accuracy: 0.4292
Color Dependency:   0.1485 (25.7%)
"""


