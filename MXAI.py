from data import get_dataset, deencode_types, image_unpreprocess
from train_model import load_model
import torch
import matplotlib.pyplot as plt
import shap
import torch.nn as nn
import numpy as np
import os
import datetime

OUTPUT_DIR = "shap_outputs"

#very weird issue where MobileNetV2 has in place Hardtanh activation which apparantly shap does not like :) this fixes it though
def disable_inplace(module):
    if isinstance(module, (nn.ReLU, nn.ReLU6, nn.Hardtanh)):
        module.inplace = False
        
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def get_two_label_samples(num_samples=5):
    device = get_device()
    
    cnn, tab_net, classifier = load_model(device=device)
    cnn.apply(disable_inplace)
    _, test_loader = get_dataset()
    
    samples = []
    found = 0
    
    for images, stats, labels in test_loader:
        images, stats, labels = images.to(device), stats.to(device), labels.to(device)
        
        with torch.no_grad():
            img_feats = cnn(images)
            tab_feats = tab_net(stats)
            logits = classifier(torch.cat((img_feats, tab_feats), dim=1))
            preds = (torch.sigmoid(logits) > 0.5).float()
            
        for i in range(images.size(0)):
            if preds[i].sum() == 2:
                samples.append((
                    images[i].unsqueeze(0),
                    stats[i].unsqueeze(0),
                    preds[i],
                    labels[i]
                ))
                found += 1
            if found >= num_samples:
                return samples
    return samples

class FullModel(nn.Module):
    def __init__(self, cnn, tab_net, classifier):
        super().__init__()
        self.cnn, self.tab_net, self.classifier = cnn, tab_net, classifier
    def forward(self, img, stat):
        img_feat = self.cnn(img)
        tab_feat = self.tab_net(stat)
        feat = torch.cat((img_feat, tab_feat), dim=1)
        return torch.sigmoid(self.classifier(feat))

def shap_analysis(num_images, num_bg_samples):
    device = get_device()

    cnn, tab_net, classifier = load_model(device=device)
    cnn.apply(disable_inplace) 
    _, test_loader = get_dataset()
    all_types = deencode_types()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    target_samples = get_two_label_samples(num_samples=num_images)
    
    backgrounds_img = []
    backgrounds_stat = []
    bg_count = 0
    
    for images, stats, _ in test_loader:
        images, stats = images.to(device), stats.to(device)
        for i in range(images.size(0)):
            backgrounds_img.append(images[i:i+1])
            backgrounds_stat.append(stats[i:i+1])
            bg_count += 1
            if bg_count >= num_bg_samples:
                break
        if bg_count >= num_bg_samples:
            break

    full_model = FullModel(cnn, tab_net, classifier).to(device)
    full_model.eval()

    # enable gradients for all parameters
    for param in full_model.parameters():
        param.requires_grad_(True)

    bg = [torch.cat(backgrounds_img), torch.cat(backgrounds_stat)]

    for sample_idx, (img, stat, pred, true_label) in enumerate(target_samples):
        
        explainer = shap.GradientExplainer(full_model, bg)
        shap_vals = explainer.shap_values([img, stat])

        img_np = image_unpreprocess(img[0].cpu()).permute(1, 2, 0).numpy()
        pred_classes = [i for i, v in enumerate(pred.cpu().numpy()) if v]
        true_classes = [i for i, v in enumerate(true_label.cpu().numpy()) if v] 

        for class_idx in pred_classes:
            if len(shap_vals) > 1:
                shap_img_vals = shap_vals[0][0]
            else:
                shap_img_vals = shap_vals[0]
                
            # no idea why, but the output is in different shapes, this works.
            if len(shap_img_vals.shape) == 4:
                print(shap_img_vals.shape)
                shap_img_class = shap_img_vals[:, :, :, class_idx]
            elif len(shap_img_vals.shape) == 3:
                print(shap_img_vals.shape)
                shap_img_class = shap_img_vals


            shap_img_class = np.transpose(shap_img_class, (1, 2, 0))

            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # original
            ax1.imshow(img_np)
            ax1.set_title(f'Original Image - Sample {sample_idx + 1}')
            ax1.axis('off')
            
            #SHAP overlay
            ax2.imshow(img_np)
            shap_sum = np.sum(shap_img_class, axis=2)
            
            # center around 0 
            vmax = np.abs(shap_sum).max()
            
            # changed the code to use the negative values as well, but it also gets a bit less clear since the "base" colour at 0 is white
            im = ax2.imshow(shap_sum, cmap='seismic', alpha=0.7, vmin=-vmax, vmax=vmax)
            
            true_types_str = ", ".join([all_types[i] for i in true_classes])
            pred_types_str = ", ".join([all_types[i] for i in pred_classes])
            ax2.set_title(f'SHAP Overlay for class "{all_types[class_idx]}"\nTrue: {true_types_str} | Pred: {pred_types_str}')
            ax2.axis('off')
            
            plt.colorbar(im, ax=ax2)
            
            plt.tight_layout()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(OUTPUT_DIR, f"{timestamp}_shap_sample_{sample_idx}_class_{class_idx}_bg_{num_bg_samples}.png"), dpi=150, bbox_inches='tight')
            plt.close()
    
if __name__ == "__main__":
    shap_analysis(num_images=1, num_bg_samples=1)