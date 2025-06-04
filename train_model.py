import torch.nn as nn
import torchvision.models as models
import torch

#i really hope im doing something correct with this
def build_model(tab_dim):
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
        nn.Linear(256, XXXX) #TODO
    )

    return cnn, tab_net, classifier


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn, tab_net, classifier = build_model()
    cnn.to(device)
    tab_net.to(device)
    classifier.to(device)

