import torch.nn as nn
import torchvision.models as models
import torch
import torch.optim as optim
from tqdm import tqdm
from data import get_dataset
from data import deencode_types, image_unpreprocess
import matplotlib.pyplot as plt

#i really hope im doing something correct with this
def build_model(tab_dim=35): #might change
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


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    train_loader, test_loader = get_dataset()
    
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
    num_epochs = 5

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


        evaluate(cnn, tab_net, classifier, test_loader, device)

    torch.save({'cnn_state_dict': cnn.state_dict(),'tab_net_state_dict': tab_net.state_dict(),'classifier_state_dict': classifier.state_dict(),}, 'pokemon_model.pt')

#Accuracy is how well it does for guessing both types
def evaluate(cnn, tab_net, classifier, test_loader, device):

    cnn.eval()
    tab_net.eval()
    classifier.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for images, stats, labels in test_loader:
            images, stats, labels = images.to(device), stats.to(device), labels.to(device)

            image_feats = cnn(images)
            tab_feats = tab_net(stats)
            combined = torch.cat((image_feats, tab_feats), dim=1)
            outputs = classifier(combined)

            #logits to predicted classes
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).float().mean().item()
            total += 1

    print(f"accuracy: {correct / total:.4f}")


#this being a multimodal model makes loading a bit harder 
def load_model(tab_dim, path='pokemon_model.pt', device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


#train()

def show_predictions(device=None, num_images=6):
    """
    Displays a few Pokémon with their predicted vs. actual types.
    
    Args:
      cnn, tab_net, classifier: your trained model components
      test_loader: DataLoader for your test split
      device: torch.device (defaults to CUDA if available)
      num_images: how many examples to show
    """
    train_loader, test_loader = get_dataset() #this is probably not correct but it works for testing it now
    cnn, tab_net, classifier = load_model()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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


    show_predictions()