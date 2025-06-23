import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from train_model import load_model, get_device

"""
    single sample explanation 
    -Grad-CAM saliency heatmap
    -Gradient contribution for tabular input
"""
def visualize_sample(cnn, tab_net, classifier, image, tabular, label_names, true_label=None, device=None):
    if device is None:
        device = get_device()

    cnn.eval()
    tab_net.eval()
    classifier.eval()

    image = image.unsqueeze(0).to(device)
    tabular = tabular.unsqueeze(0).to(device)

    image.requires_grad = True
    tabular.requires_grad = True

    img_feat = cnn(image)
    tab_feat = tab_net(tabular)
    combined = torch.cat((img_feat, tab_feat), dim=1)
    output = classifier(combined)
    probs = torch.sigmoid(output)
    pred_vec = (probs > 0.5).float()[0]

    top_idx = torch.topk(probs[0], 1).indices.item()

    #gradient for class
    classifier.zero_grad()
    cnn.zero_grad()
    tab_net.zero_grad()
    output[0, top_idx].backward(retain_graph=True)

    #tab gradient map
    tab_grad = tabular.grad.abs().squeeze().cpu().numpy()
    tab_importance = np.mean(tab_grad)

    #img gradient map
    image_grad = image.grad.squeeze().abs().cpu().mean(dim=0).numpy()
    img_importance = np.mean(image_grad)
    heatmap = (image_grad - image_grad.min()) / (image_grad.max() - image_grad.min() + 1e-8)

    from data import image_unpreprocess
    img_disp = image_unpreprocess(image.squeeze().detach().cpu())
    img_disp = img_disp.permute(1, 2, 0).clip(0, 1).numpy()

    #lets just pray it work
    fig, axs = plt.subplots(1, 2, figsize=(15, 4))

    axs[0].imshow(img_disp)
    axs[0].imshow(heatmap, cmap='jet', alpha=0.5)
    axs[0].set_title("Image Gradient Map")
    axs[0].axis("off")

    axs[1].bar(["Image", "Tabular"], [img_importance, tab_importance], color=['orange', 'green'])
    axs[1].set_title("Modality Gradient Contribution")
    axs[1].set_ylim(0, max(img_importance, tab_importance) * 1.2)

    plt.tight_layout()
    plt.savefig("test_visualization.png")


from data import get_dataset, deencode_types

cnn, tab_net, classifier = load_model()
train_loader, test_loader = get_dataset()
#sample
images, stats, labels = next(iter(test_loader))
SAMPLE = 3
sample_img, sample_tab, sample_label = images[SAMPLE], stats[SAMPLE], labels[SAMPLE]
type_names = deencode_types()


visualize_sample(cnn, tab_net, classifier, sample_img, sample_tab, type_names, true_label=sample_label)
