import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from data import get_dataset, deencode_types, image_unpreprocess, create_or_load_dataframe, tab_preprocess, get_sample_by_idx
from train_model import load_model, get_device
from XAI_hue_shift_analysis import hue_shift_transform

#hue script noise loop but for local
def compute_noise_threshold(explain_fn, img, tab, data_std, target_fn, target_value, cmp_op, step=0.01, max_noise=3.0):
    img0 = img.clone()
    base_exp = explain_fn(img0) if tab is None else explain_fn(img0, tab.clone())

    σ = step
    while σ <= max_noise:
        noise = torch.randn_like(img) * (σ * data_std)
        img_n = torch.clamp(img + noise, -3, 3)
        if tab is None:
            exp_n = explain_fn(img_n)
        else:
            exp_n = explain_fn(img_n, tab)
        metric = target_fn(base_exp, exp_n)
        if cmp_op(metric, target_value):
            return σ
        σ += step
    return None


#Saliency continuity
def saliency_explain(cnn, input_tensor):
    x = input_tensor.clone().detach().requires_grad_(True)
    cnn.zero_grad()
    logits = cnn(x)
    score  = logits.max(1)[0]
    score.backward()
    saliency_map = x.grad.abs().squeeze().mean(dim=0).cpu().numpy()

    return (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

def saliency_metric(s0, s1):
#Pearson correlation
    return pearsonr(s0.flatten(), s1.flatten())[0]

#Contribution continuity
def contrib_explain(cnn, tab_net, classifier, img, tab):
    cnn.eval()
    tab_net.eval()
    classifier.eval()

    img = img.clone().detach().requires_grad_(True)
    tab = tab.clone().detach().requires_grad_(True)

    #forward pass
    img_feats = cnn(img)
    tab_feats = tab_net(tab)
    output = classifier(torch.cat([img_feats, tab_feats], dim=1))
    score = output.max(1)[0]
    score.backward()

    #mean absolute gradients
    img_imp = img.grad.abs().mean().item()
    tab_imp = tab.grad.abs().mean().item()
    return img_imp / (img_imp + tab_imp)



device = get_device()
cnn, tab_net, classifier = load_model(device=device)
cnn.to(device); tab_net.to(device); classifier.to(device)

# pick one Pokemon
df = create_or_load_dataframe()
df = tab_preprocess(df)
img, tab, lbl = get_sample_by_idx(df, 0, hue_shift_transform(0))
img = img.unsqueeze(0).to(device).requires_grad_(True)
tab = tab.unsqueeze(0).to(device).requires_grad_(True)
_, test_loader = get_dataset()
all_images = []
for images, _, _ in test_loader:
    all_images.append(images.numpy())
data_std = np.std(np.vstack(all_images))

#saliency
sal_thresh = compute_noise_threshold(
    explain_fn=lambda i: saliency_explain(cnn, i),
    img=img, tab=None,
    data_std=data_std,
    target_fn=saliency_metric, target_value=0.90,
    cmp_op=lambda m,th: m < th
)
#contribution
contrib_thresh = compute_noise_threshold(
    explain_fn=lambda i,t: contrib_explain(cnn, tab_net, classifier, i, t),
    img=img, tab=tab,
    data_std=data_std,
    target_fn=lambda base,new: abs(new-base),
    target_value=0.05,
    cmp_op=lambda delta,th: delta > th
)

print(f"Saliency continuity threshold: {sal_thresh:.3f}  (ρ drop < 0.90)")
print(f"Contribution continuity threshold: {contrib_thresh:.3f}  (delta r > 0.05)")