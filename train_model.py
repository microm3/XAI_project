import torch.nn as nn
import torchvision.models as models
import torch
import torch.optim as optim
from tqdm import tqdm
from data import get_dataset
from data import deencode_types, image_unpreprocess
import matplotlib.pyplot as plt


train_loader, test_loader = get_dataset()
TAB_DIM = 17

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


#i really hope im doing something correct with this
def build_model(tab_dim=TAB_DIM): #might change
    #MobileNetV2 but remove the last layer # this is a thing i learned in the datascience course is pretty cool was super confused bout it at first lol
    cnn = models.mobilenet_v2(pretrained=True)
    cnn.classifier = nn.Identity()

    #we can change this,, but idk basic fully connected dense neural network
    #tab_dim is like the input layer which is number of stats
    #64 unit hidden layer, just 1 should be enough i think
    #then 64 out but with no activation as we dont wanna make predictions? 
    tab_net = nn.Sequential(
        nn.Linear(tab_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64)
    )

    #combination classifier
    #combines the both, 1280 is the feature vector size that mobile net v2 outputs + the 64 from the tabular data
    #relu to make it not so linear 
    #output makes it to the number of different classes we output which should be a fixed number but idk what
    #also dunno how we make this multi class
    classifier = nn.Sequential(
        nn.Linear(1280 + 64, 256),
        nn.ReLU(),
        nn.Linear(256, 18)
    )

    return cnn, tab_net, classifier


def train(num_epochs=150):
    device = get_device()
    
    #some useful data maybe
    sample_img, sample_tab, sample_label = next(iter(train_loader))[0][0], next(iter(train_loader))[1][0], next(iter(train_loader))[2][0]
    tab_dim = sample_tab.shape[0]
    num_types = sample_label.shape[0]

    #build up model
    cnn, tab_net, classifier = build_model()
    cnn.to(device)
    tab_net.to(device)
    classifier.to(device)

    #optimizer and loss function
    params = list(cnn.parameters()) + list(tab_net.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=1e-4)
    #one hot vector uses this apparantly
    criterion = nn.BCEWithLogitsLoss()

    
    #training, sigmoid + binary cross-entropy for each type
    
    best_acc = 0
    stagnant_epochs = 0
    acc_list = []
    epoch_list = []

    for epoch in range(num_epochs):
        cnn.train()
        tab_net.train()
        classifier.train()
        running_loss = 0.0

        for images, stats, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, stats, labels = images.to(device), stats.to(device), labels.to(device)

            optimizer.zero_grad()

            image_feats = cnn(images)
            tab_feats = tab_net(stats)
            #concatinate images and feats into the combined is what internet told me
            combined = torch.cat((image_feats, tab_feats), dim=1)
            outputs = classifier(combined)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        assert not torch.isnan(outputs).any(), "output has the NaN"
        assert not torch.isnan(labels).any(), "labels has the NaN"
        print(f"epoch {epoch+1}, has loss: {running_loss / len(train_loader):.4f}")

        if epoch % 5 == 0:
                acc = evaluate(cnn, tab_net, classifier, test_loader, device)
                acc_list.append(acc)
                epoch_list.append(epoch)

                if acc > best_acc:
                    best_acc = acc
                    stagnant_epochs = 0
                else:
                    stagnant_epochs += 5

                if stagnant_epochs >= 30:
                    print("Early stopping triggered.")
                    break
        
    plt.plot(epoch_list, acc_list, marker='o')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('accuracy_curve_comb.png')

    torch.save({'cnn_state_dict': cnn.state_dict(),'tab_net_state_dict': tab_net.state_dict(),'classifier_state_dict': classifier.state_dict(),}, 'pokemon_model.pt')



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
#Accuracy is how well it does for guessing both types
def evaluate(cnn, tab_net, classifier, test_loader, device = None):

    if device is None:
        device = get_device()
    cnn.eval()
    tab_net.eval()
    classifier.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, stats, labels in test_loader:
            images, stats, labels = images.to(device), stats.to(device), labels.to(device)

            image_feats = cnn(images)
            tab_feats = tab_net(stats)
            combined = torch.cat((image_feats, tab_feats), dim=1)
            outputs = classifier(combined)

            #logits to predicted classes
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    avg_type = 'binary' if all_labels.shape[1] == 1 or len(all_labels.shape) == 1 else 'macro'


    label_accuracy = (all_preds == all_labels).sum() / all_labels.size
    print(f"Per-label accuracy: {label_accuracy:.4f}")

    print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds, average=avg_type):.4f}")
    print(f"Recall:    {recall_score(all_labels, all_preds, average=avg_type):.4f}")
    print(f"F1-score:  {f1_score(all_labels, all_preds, average=avg_type):.4f}")

    try:
        print(f"AUC-ROC:   {roc_auc_score(all_labels, all_preds, average=avg_type):.4f}")
    except ValueError:
        print("AUC-ROC:   Not defined (possibly only one class present)")

  
    return accuracy_score(all_labels, all_preds)


#this being a multimodal model makes loading a bit harder 
def load_model(tab_dim=TAB_DIM, path='pokemon_model.pt', device=None):
    if device is None:
        device = get_device()

    #do empty model
    cnn, tab_net, classifier = build_model(tab_dim)

    #load the dicts
    checkpoint = torch.load(path, map_location=device)
    cnn.load_state_dict(checkpoint['cnn_state_dict'])
    tab_net.load_state_dict(checkpoint['tab_net_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    cnn.to(device)
    tab_net.to(device)
    classifier.to(device)

    cnn.eval()
    tab_net.eval()
    classifier.eval()

    print("model loaded from save")
    
    return cnn, tab_net, classifier


if __name__ == "__main__":
    train(500)
    cnn, tab_net, classifier = load_model()
    evaluate(cnn, tab_net, classifier, test_loader)
    # not sure why device is passed as a hard coded argument here. Letting it be a nullable argument in case needed for Janine's computer.
    # evaluate(cnn, tab_net, classifier, test_loader, torch.device("cuda"))



def show_predictions(device=None, num_images=6):
    """
    Displays a few Pokémon with their predicted vs. actual types.
    
    Args:
      cnn, tab_net, classifier: your trained model components
      test_loader: DataLoader for your test split
      device: torch.device (defaults to CUDA if available)
      num_images: how many examples to show
    """
    #train_loader, test_loader = get_dataset() #this is probably not correct but it works for testing it now
    cnn, tab_net, classifier = load_model()

    if device is None:
        device = get_device()
    
    all_types = deencode_types()  # list of type names in order
    cnn.to(device).eval()
    tab_net.to(device).eval()
    classifier.to(device).eval()

    images_shown = 0
    plt.figure(figsize=(12, num_images * 2))
    
    with torch.no_grad():
        for images, stats, labels in test_loader:
            images, stats, labels = images.to(device), stats.to(device), labels.to(device)
            img_feats = cnn(images)
            tab_feats = tab_net(stats)
            logits   = classifier(torch.cat((img_feats, tab_feats), dim=1))
            preds    = (torch.sigmoid(logits) > 0.5).cpu().float()

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break

                img = image_unpreprocess(images[i].cpu()).permute(1,2,0).clip(0,1)
                true_vec = labels[i].cpu()
                pred_vec = preds[i]

                true_types = [all_types[j] for j in true_vec.nonzero().flatten().tolist()]
                pred_types = [all_types[j] for j in pred_vec.nonzero().flatten().tolist()]

                ax = plt.subplot(num_images//3, 3, images_shown+1)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f"True: {','.join(true_types) or 'None'}\n"
                             f"Pred: {','.join(pred_types) or 'None'}",
                             fontsize=10)

                images_shown += 1

            if images_shown >= num_images:
                break

    plt.tight_layout()
    plt.savefig("pred_example.png")


#show_predictions()




"""
old, but this one is a bit suspect.. 
Per-label accuracy: 0.9943
Accuracy:  0.9304
Precision: 0.9779
Recall:    0.9506
F1-score:  0.9638
AUC-ROC:   0.9743
"""




"""
new data code, with seed
it stopped early, but the accuracy is sh#t 

Per-label accuracy: 0.9114
Accuracy:  0.1996
Precision: 0.4289
Recall:    0.3228
F1-score:  0.3473
AUC-ROC:   0.6426


Janines try
Per-label accuracy: 0.9653
Accuracy:  0.5916
Precision: 0.8371
Recall:    0.7209
F1-score:  0.7716
AUC-ROC:   0.8538


"""

