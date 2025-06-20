import torch
from torchvision import transforms
from data import get_dataset
from train_model_iso_img import load_model, evaluate
import warnings
from data import add_white_background
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

def create_grayscale_transform():
    return transforms.Compose([
        transforms.Lambda(add_white_background),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def analyze_color_dependency():
    device = get_device()
    
    cnn, classifier = load_model(path=MODEL_PATH, device=device)
    print(f"model loaded from {MODEL_PATH}")
    
    _, rgb_test_loader = get_dataset()
    print("Testing on RGB images:")
    rgb_accuracy = evaluate(cnn, classifier, rgb_test_loader, device)
    
    grayscale_transform = create_grayscale_transform()
    _, gray_test_loader = get_dataset(image_transform=grayscale_transform)
    print("\nTesting on grayscale images:")
    gray_accuracy = evaluate(cnn, classifier, gray_test_loader, device)
    
    color_dependency = rgb_accuracy - gray_accuracy
    color_dependency_pct = (color_dependency / rgb_accuracy) * 100
    
    print(f"RGB Accuracy:      {rgb_accuracy:.4f}")
    print(f"Grayscale Accuracy: {gray_accuracy:.4f}")
    print(f"Color Dependency:   {color_dependency:.4f} ({color_dependency_pct:.1f}%)")

if __name__ == "__main__":
    analyze_color_dependency()
    
    
    
"""
new model with seed 
Testing on RGB images:
Per-label accuracy: 0.9640
Accuracy:  0.5777
Precision: 0.8379
Recall:    0.6895
F1-score:  0.7504
AUC-ROC:   0.8385

Testing on grayscale images:
Per-label accuracy: 0.9223
Accuracy:  0.2030
Precision: 0.6184
Recall:    0.3739
F1-score:  0.4279
AUC-ROC:   0.6740
RGB Accuracy:      0.5777
Grayscale Accuracy: 0.2030
Color Dependency:   0.3747 (64.9%)
"""
