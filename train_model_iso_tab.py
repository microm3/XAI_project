import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data import get_dataset
import matplotlib.pyplot as plt

# Default dimensions (will infer from data if None)
TAB_DIM = None
NUM_CLASSES = 18
train_loader, test_loader = get_dataset()

# Tabular-only model builder
def build_model(tab_dim, num_classes):
    """
    Builds a simple tabular-only model:
    - tab_net: processes input features (tab_dim) into latent (64)
    - classifier: maps latent to num_classes
    """
    tab_net = nn.Sequential(
        nn.Linear(tab_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU()
    )

    classifier = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

    return tab_net, classifier

# Training loop
def train():
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load data (assumes get_dataset returns (imgs, stats, labels), ignore imgs)
 

    # Infer dims
    _, sample_tab, sample_label = next(iter(train_loader))
    tab_dim = sample_tab.shape[1]
    num_classes = sample_label.shape[1]
    print(f"Tabular input dim: {tab_dim}, #Classes: {num_classes}")

    # Build model
    tab_net, classifier = build_model(tab_dim, num_classes)
    tab_net.to(device)
    classifier.to(device)

    # Optimizer and loss
    optimizer = optim.Adam(
        list(tab_net.parameters()) + list(classifier.parameters()),
        lr=1e-4
    )
    criterion = nn.BCEWithLogitsLoss()

    # Train
    num_epochs = 1000
    best_acc = 0
    stagnant_epochs = 0
    acc_list = []
    epoch_list = []
    for epoch in range(1, num_epochs+1):
        tab_net.train()
        classifier.train()
        total_loss = 0.0

        for _, stats, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            stats = stats.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            feats = tab_net(stats)
            outputs = classifier(feats)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        if epoch % 5 == 0:
            acc = evaluate(tab_net, classifier, test_loader, device)
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

    # Save checkpoint
    torch.save({
        'tab_net': tab_net.state_dict(),
        'classifier': classifier.state_dict(),
    }, 'pokemon_model_tabular.pt')
    print("Saved tabular model to pokemon_model_tabular.pt")

    plt.plot(epoch_list, acc_list, marker='o')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('accuracy_curve_tab.png')


    # Eval
    tab_net.eval()
    classifier.eval()
    evaluate(tab_net, classifier, test_loader, device)


def evaluate(tab_net, classifier, test_loader, device):
    all_preds, all_labels = [], []

    with torch.no_grad():
        for _, stats, labels in test_loader:
            stats = stats.to(device)
            labels = labels.to(device).float()

            feats = tab_net(stats)
            logits = classifier(feats)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    avg_type = 'macro'

    print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds, average=avg_type, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(all_labels, all_preds, average=avg_type):.4f}")
    print(f"F1-score:  {f1_score(all_labels, all_preds, average=avg_type):.4f}")
    try:
        print(f"AUC-ROC:   {roc_auc_score(all_labels, all_preds, average=avg_type):.4f}")
    except Exception:
        print("AUC-ROC:   n/a")
    
    return accuracy_score(all_labels, all_preds)


def load_model(path='pokemon_model_tabular.pt', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # We need to know tab_dim and num_classes; load dummy data
    train_loader, _ = get_dataset()
    _, sample_tab, sample_label = next(iter(train_loader))
    tab_dim = sample_tab.shape[1]
    num_classes = sample_label.shape[1]

    tab_net, classifier = build_model(tab_dim, num_classes)
    checkpoint = torch.load(path, map_location=device)
    tab_net.load_state_dict(checkpoint['tab_net'])
    classifier.load_state_dict(checkpoint['classifier'])
    tab_net.to(device)
    classifier.to(device)
    tab_net.eval()
    classifier.eval()
    print("Loaded tabular model from", path)
    return tab_net, classifier


train()
tab_net, classifier = load_model()
evaluate(tab_net, classifier, test_loader, device="cuda")