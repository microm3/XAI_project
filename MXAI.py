from data import get_dataset, deencode_types, image_unpreprocess
from train_model import load_model
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import shap
import torch.nn as nn
import numpy as np


#very weird issue where MobileNetV2 has in place Hardtanh activation which apparantly shap does not like :) this fixes it though
def disable_inplace(module):
    if isinstance(module, (nn.ReLU, nn.ReLU6, nn.Hardtanh)):
        module.inplace = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn, tab_net, classifier = load_model(device=device)
cnn.apply(disable_inplace) 
_, test_loader = get_dataset()
all_types = deencode_types()




#want some images that have 2 classes specifically, to test the eh youknow difference in class heatmap?
def twolabel_sample():
    for images, stats, labels in test_loader:
        images, stats, labels = images.to(device), stats.to(device), labels.to(device)
        with torch.no_grad():
            img_feats = cnn(images)
            tab_feats = tab_net(stats)
            logits = classifier(torch.cat((img_feats, tab_feats), dim=1))
            preds = (torch.sigmoid(logits) > 0.5).float()
        for i in range(images.size(0)):
            if preds[i].sum() == 2:
                return images[i].unsqueeze(0), stats[i].unsqueeze(0), preds[i]

img, stat, pred = twolabel_sample()
print("Selected samples: ", [all_types[i] for i, v in enumerate(pred.cpu().numpy()) if v])


class FullModel(nn.Module):
    def __init__(self, cnn, tab_net, classifier):
        super().__init__()
        self.cnn, self.tab_net, self.classifier = cnn, tab_net, classifier
    def forward(self, img, stat):
        img_feat = self.cnn(img).clone()
        tab_feat = self.tab_net(stat).clone()
        feat = torch.cat((img_feat, tab_feat), dim=1)
        return torch.sigmoid(self.classifier(feat))

full_model = FullModel(cnn, tab_net, classifier).to(device)

bg = [img.detach().clone(), stat.detach().clone()]
explainer = shap.DeepExplainer(full_model, bg)
shap_vals = explainer.shap_values([img, stat])


#why is just plotting some goddman images so fing hard >:(
#shap_vals[0][0].shape: (3, 224, 224, 18) (rgb, x, y, output classes?)
img_np = image_unpreprocess(img[0].cpu()).permute(1, 2, 0).numpy()

pred_classes = [i for i, v in enumerate(pred.cpu().numpy()) if v] 

for class_idx in pred_classes:
    shap_img_class = shap_vals[0][0][..., class_idx]
    shap_img_class = np.transpose(shap_img_class, (1, 2, 0))

    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)
    plt.imshow(np.sum(np.abs(shap_img_class), axis=2), cmap='jet', alpha=0.5)
    plt.colorbar()
    plt.title(f'SHAP overlay for class "{all_types[class_idx]}"')
    plt.axis('off')
    plt.savefig(f"shap_{class_idx}.png")


