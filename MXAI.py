from data import get_dataset, deencode_types, image_unpreprocess
from train_model import build_model, load_model
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import shap


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn, tab_net, classifier = load_model(device=device)
_, test_loader = get_dataset()
all_types = deencode_types()


#want some images that have 2 classes specifically, to test the eh youknow difference in class heatmap?
def twolabel_samples():
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

img, stat, pred = twolabel_samples()
print("selected sample = ", [all_types[i] for i, v in enumerate(pred.numpy()) if v])


def shap_multimodal_explanation(img, stat, pred, background_size=50):

    bg_img = img
    bg_stat = stat.cpu().numpy()

    def model_forward_stats(x_stat):
        x_stat = torch.tensor(x_stat, device=device, dtype=stat.dtype)
        with torch.no_grad():
            feats_i = cnn(img.to(device))
            feats_t = tab_net(x_stat)
            out = classifier(torch.cat((feats_i, feats_t), dim=1))
            return torch.sigmoid(out).cpu().numpy()

    #create explainers
    explainer_img = shap.DeepExplainer((cnn,), [bg_img])
    explainer_tab = shap.KernelExplainer(model_forward_stats, bg_stat)

    shap_vals_img = explainer_img.shap_values(img)
    shap_vals_tab = explainer_tab.shap_values(stat.cpu().numpy())


    plt.figure(figsize=(6, 6))
    shap.image_plot(shap_vals_img, image_unpreprocess(img.cpu()).permute(0,2,3,1).numpy())
    shap.summary_plot(shap_vals_tab, stat.cpu().numpy(), feature_names=[f"feat_{i}" for i in range(stat.size(1))])
    plt.savefig("shapexp.png")



shap_multimodal_explanation(img, stat, pred)