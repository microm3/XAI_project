import torch
import matplotlib.pyplot as plt
import numpy as np
#import torchvision.transforms.functional as F
import torch.nn.functional as F
from train_model import load_model, get_device
from data import get_dataset, deencode_types, image_unpreprocess
import tensorflow as tf
from tf_keras_vis.saliency import Saliency

"""
    single sample explanation 
    -Grad-CAM saliency heatmap
    -Gradient contribution for tabular input
"""
def contribution(cnn, tab_net, classifier, image, tabular, label_names, true_label=None, device=None):
    if device is None:
        device = get_device()

    cnn.eval()
    tab_net.eval()
    classifier.eval()

    image = image.unsqueeze(0).to(device)
    tabular = tabular.unsqueeze(0).to(device)

    image.requires_grad = True
    tabular.requires_grad = True

    #tab gradient map
    tab_grad = tabular.grad.abs().squeeze().cpu().numpy()
    tab_importance = np.mean(tab_grad)

    #img gradient map
    image_grad = image.grad.squeeze().abs().cpu().mean(dim=0).numpy()
    img_importance = np.mean(image_grad)


    #normalize
    total_importance = img_importance + tab_importance
    img_contrib = img_importance / total_importance
    tab_contrib = tab_importance / total_importance



    from data import image_unpreprocess
    img_disp = image_unpreprocess(image.squeeze().detach().cpu())
    img_disp = img_disp.permute(1, 2, 0).clip(0, 1).numpy()

    #lets just pray it work
    fig, axs = plt.subplots(1, 1, figsize=(15, 4))

    axs[1].bar(["Image", "Tabular"], [img_contrib, tab_contrib], color=['orange', 'green'])
    axs[1].set_title("Modality Gradient Contribution")
    axs[1].set_ylim(0, 1.0)
    axs[1].text(0, img_contrib + 0.02, f"{img_contrib:.2%}", ha='center')
    axs[1].text(1, tab_contrib + 0.02, f"{tab_contrib:.2%}", ha='center')

    plt.tight_layout()
    plt.savefig("test_visualization.png")



def compute_saliency_map(model, input_tensor):
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    model.zero_grad()
    output = model(input_tensor)
    score, _ = output.max(dim=1)
    score.backward()
    saliency, _ = input_tensor.grad.abs().max(dim=1)
    saliency = saliency.squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency


def compute_SmoothGrad(model, input_tensor, n_samples=25, noise_level=0.1):
    input_tensor = input_tensor.clone().detach()
    avg_grad = torch.zeros_like(input_tensor)
    for _ in range(n_samples):
        noise = torch.randn_like(input_tensor) * noise_level
        noisy = (input_tensor + noise).requires_grad_(True)
        output = model(noisy)
        score = output.max(dim=1)[0]
        model.zero_grad()
        score.backward()
        avg_grad += noisy.grad.abs()
    saliency = avg_grad.squeeze().mean(dim=0).cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency


if __name__ == '__main__':
    device = get_device()
    cnn, tab_net, classifier = load_model(device=device)
    cnn.to(device); tab_net.to(device); classifier.to(device)
    _, test_loader = get_dataset()
    images, stats, labels = next(iter(test_loader))
    img = images[0].unsqueeze(0).to(device)

    #Saliency
    sal = compute_saliency_map(cnn, img)
    plt.imsave('saliency_map.png', sal, cmap='jet')

    #SmoothGrad
    smooth = compute_SmoothGrad(cnn, img)
    plt.imsave('smoothgrad_map.png', smooth, cmap='jet')


    #Modality contribution
    contribution(cnn, tab_net, classifier,
                 images[0], stats[0], deencode_types())

