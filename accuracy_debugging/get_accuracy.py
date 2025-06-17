import torch
import sys
import os

# absolutely wanted a separate directory  
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

os.chdir(parent_dir)
sys.path.append(parent_dir)
from data import get_dataset
from train_model import load_model, evaluate


MODEL_PATH = os.path.join(script_dir, 'reproduce_bug_model_150epochs.pt')
# MODEL_PATH = 'pokemon_model.pt' 

def get_accuracies():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"using device: {device}")
    
    cnn, tab_net, classifier = load_model(path=MODEL_PATH, device=device)
    _, test_loader = get_dataset()
    
    exact_acc = evaluate(cnn, tab_net, classifier, test_loader, device)
    
    return exact_acc

if __name__ == "__main__":
    get_accuracies() 
    
    
    
# NOTE: loading the newly trained model, it shows the same accuracy as the training curve. So the setups seems to work. 
# The original model however, shows a different accuracy than the one plotted, not sure why that is.. 


"""
pokemon_model.pt
Per-label accuracy: 0.9943
Accuracy:  0.9304  # weird.
Precision: 0.9779
Recall:    0.9506
F1-score:  0.9638
AUC-ROC:   0.9743
"""

"""
reproduce_bug_model_150epochs.pt
Per-label accuracy: 0.9636
Accuracy:  0.5661 #! 
Precision: 0.8201
Recall:    0.6946
F1-score:  0.7496
AUC-ROC:   0.8405
"""