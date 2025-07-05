import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from train_model_iso_img import load_model, get_device
from XAI_hue_shift_analysis import MODEL_PATH, hue_shifts, hue_shift_transform
from data import get_sample_by_idx, deencode_types, get_dataset
from continuity_testing_hue_shift import evaluate_with_noise, find_noise_threshold
import matplotlib.pyplot as plt
from data import create_or_load_dataframe, tab_preprocess

device = get_device()
cnn, classifier = load_model(path=MODEL_PATH, device=device)

#global data_std
_, test_loader = get_dataset()
all_images = []
for images, _, _ in test_loader:
    all_images.append(images.numpy())
data_std = np.std(np.vstack(all_images))


#fake dataset thingy with only 1 pokemon in there but it works
class SinglePokemon(Dataset):
    def __init__(self, image, stats, label):
        self.image = image.unsqueeze(0)
        self.stats = stats.unsqueeze(0)
        self.label = label.unsqueeze(0)
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.image[0], self.stats[0], self.label[0]


#bulbasaur <3
sample_idx = 0
df = create_or_load_dataframe()
df = tab_preprocess(df)

results = []

for shift_name, shift_deg in hue_shifts.items():
    transform = hue_shift_transform(shift_deg)
    image, stats, label = get_sample_by_idx(df, sample_idx, transform)
    
    dataset = SinglePokemon(image, stats, label)
    loader = DataLoader(dataset, batch_size=1)
    
    pred_logits = classifier(cnn(image.unsqueeze(0).to(device)))
    probs = torch.sigmoid(pred_logits)
    pred = (probs > 0.5).float().squeeze(0).cpu()
    pred_types = [deencode_types()[i] for i in pred.nonzero(as_tuple=True)[0].tolist()]

    #resuse from global, to have same concept
    baseline_acc = evaluate_with_noise(cnn, classifier, loader, device, 0.0, data_std)
    threshold = find_noise_threshold(baseline_acc, cnn, classifier, loader, device, data_std)

    results.append((shift_name, shift_deg, pred_types, image, threshold))

print(f"\n{'Shift Name':<15} {'Deg':<5} Predicted Types         Threshold")
print("-" * 60)
for name, deg, preds, _, thresh in results:
    pred_str = ', '.join(preds) if preds else '—'
    print(f"{name:<15} {deg:<5} {pred_str:<25} {thresh:.3f}" if thresh else f"{name:<15} {deg:<5} {pred_str:<25} None")

def denorm(tensor_img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(tensor_img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(tensor_img.device)
    return torch.clamp(tensor_img * std + mean, 0, 1)

n = len(results)
cols = 4
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
axes = axes.flatten()
for ax, (name, deg, preds, img, _) in zip(axes, results):
    ax.imshow(denorm(img).permute(1,2,0).cpu())
    ax.axis('off')
    ax.set_title(f"{name}\n{deg}° → {', '.join(preds)}", fontsize=8)
for ax in axes[n:]:
    ax.axis('off')
plt.tight_layout()
plt.savefig("individual_continuity_hueshift.png")