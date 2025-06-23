#create dataset combine from both images + stats ---> get_data returns (image_tensor, stats_tensor, label_tensor)
import os
import pandas as pd
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import torch
from PIL import Image
import re
from torch.utils.data import Dataset
from torch.utils.data import random_split
from overlap_analysis import create_overlap_analysis_csv

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import to_hex
from sklearn.decomposition import PCA

# refactored to be used in shap as well - and not risk "dropping" different ones
def get_excluded_columns():
    df_sample = pd.read_csv('pokemon.csv')
    
    excluded = ["name", "type1", "type2", "japanese_name", "classfication", "abilities"]    
    against_cols = [col for col in df_sample.columns if col.startswith("against_")]
    
    return excluded + against_cols

def normalize(name):
    return re.sub(r'[\s\-_]', '', name.lower())

def clean_stat(val):
    if isinstance(val, str):
        #sometimes python is shit
        match = re.search(r'\d+', val)
        if match:
            return float(match.group(0))
        else:
            return 0.0 
    return float(val)

def create_or_load_dataframe():
    pickle_path = "combined_dataset.pkl"
    if os.path.exists(pickle_path):
        return pd.read_pickle(pickle_path)

    csv_path = "pokemon.csv"
    df = pd.read_csv(csv_path)

    all_types = sorted(set(df['type1']) | set(df['type2'].dropna()))
    
    # normalize names just in case
    df['name'] = df['name'].apply(normalize)

    image_root = "pokemon_sprites/"
    available_folders = {normalize(folder): folder for folder in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, folder))}

    dataset = []

    for _, row in df.iterrows():
        pokemon_name = row['name']  # already normalized
        if pokemon_name not in available_folders:
            print(f"Missing: {pokemon_name}")
            continue

        folder_name = available_folders[pokemon_name]
        pokemon_folder = os.path.join(image_root, folder_name)

        for fname in os.listdir(pokemon_folder):
            full_image_path = os.path.join(pokemon_folder, fname)

            dataset.append({
                "image_path": full_image_path,
                "pokemon_name": pokemon_name,  # ADD THIS - for proper Pokemon-based splitting
                
                # store encoded types in dataframe, instead of encoding for each sample
                "encoded_types": encode_types(row["type1"], row["type2"], all_types),
                
                # keep raw types for reference/debugging if needed
                "type1": row["type1"],
                "type2": row["type2"],
                
                # TODO abilities is not included (for now), either include later or comment in report on why it was not.
                "stats": {k: clean_stat(v) for k, v in row.drop(get_excluded_columns()).items()}
            })

    create_overlap_analysis_csv(df, image_root)

    combined_df = pd.DataFrame(dataset)
    combined_df.to_pickle(pickle_path)  #pickleee
    return combined_df


def add_white_background(image):
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
        return background
    else:
        return image.convert('RGB')

def image_preprocess():
    return transforms.Compose([
        # add white background for pngs without background (otherwise they get assigned a black background)
        transforms.Lambda(add_white_background),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

def image_unpreprocess(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return tensor * std + mean


#got fucking damnit the stats were strings :() so it set them all to 0 without complaining...... types in python fun
def tab_preprocess(df):
    combined_df = df
    stats_matrix = np.stack([[v for v in s.values()] for s in combined_df['stats']])

    # print("Unique stat rows:", np.unique(stats_matrix, axis=0).shape[0])
    # print(stats_matrix[0])
    #chatgpt told me this is a thing
    scaler = StandardScaler()
    scaled_stats = scaler.fit_transform(stats_matrix)
    scaled_stats = np.nan_to_num(scaled_stats, nan=0.0, posinf=0.0, neginf=0.0)
    combined_df["scaled_stats"] = [row for row in scaled_stats]
    # print(scaled_stats[0])
    #seems usefull to save
    joblib.dump(scaler, "tabular_scaler.pkl")
    return combined_df

#types to vectors --> one hot vector 
def encode_types(type1, type2, all_types):
    vec = torch.zeros(len(all_types))
    if pd.notna(type1):
        vec[all_types.index(type1)] = 1
    if pd.notna(type2):
        vec[all_types.index(type2)] = 1
    return vec


def deencode_types():
    return sorted(set(pd.read_pickle("combined_dataset.pkl")["type1"]) |
                   set(pd.read_pickle("combined_dataset.pkl")["type2"].dropna()))

#samples a single image + types + stats
def get_sample_by_idx(df, idx, tfm):
    row = df.iloc[idx]

    image = Image.open(row["image_path"])
    
    image = tfm(image)

    tab = torch.tensor(row["scaled_stats"], dtype=torch.float32)
    label = row["encoded_types"]

    return image, tab, label

"""
Main call function
returns data in form of (image_tensor, stats_tensor, label_tensor) list (for now)
The types are kinda stupid, asked chatgpt how to do it, got like one hot vector thing but now you need deencode types to like get the types so maybe we need to change that or make an easier map
"""

#no idea what format do we want as output?
# NOTE PRIVATE 
def get_data(image_transform=None):
    df = create_or_load_dataframe()
    df = tab_preprocess(df)

    # use "old", default transform if none provided to not break old stuff
    if image_transform is None:
        image_transform = image_preprocess()

    # print(f"tab dim is : ", len(df["scaled_stats"].iloc[0]))
    data = []
    for idx in range(len(df)):
        image, stats, label = get_sample_by_idx(df, idx, image_transform)
        data.append((image, stats, label))

    return data 

class Pokemon(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, stats, label = self.data[idx]
        return image, stats, label

# version that splits by Pokemon, not by individual images
# def get_dataset():
#     df = create_or_load_dataframe()
#     df = tab_preprocess(df)
    
#     unique_pokemon = df['pokemon_name'].unique()
    
#     # split by pokemon names
#     torch.manual_seed(42)
#     pokemon_indices = torch.randperm(len(unique_pokemon))
#     train_size = int(0.8 * len(unique_pokemon))
    
#     train_pokemon = set(unique_pokemon[pokemon_indices[:train_size]])
#     test_pokemon = set(unique_pokemon[pokemon_indices[train_size:]])
    
#     print(f"Train Pokemon: {len(train_pokemon)}, Test Pokemon: {len(test_pokemon)}")
    
#     train_data = []
#     test_data = []
    
#     for idx in range(len(df)):
#         image, stats, label = get_sample_by_idx(df, idx, image_preprocess())
#         pokemon_name = df.iloc[idx]['pokemon_name']
        
#         if pokemon_name in train_pokemon:
#             train_data.append((image, stats, label))
#         else:
#             test_data.append((image, stats, label))
    
#     train_dataset = Pokemon(train_data)
#     test_dataset = Pokemon(test_data)
    
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
#     return train_loader, test_loader

# old version. potential data leakage
# NOTE PUBLIC
def get_dataset(image_transform=None):
    data = get_data(image_transform)
    dataset = Pokemon(data)

    # Use fixed seed for reproducible splits
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    train_ds, test_ds = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)
    
    return train_loader, test_loader 


from sklearn.manifold import TSNE
import plotly.express as px
import torchvision.models as models
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = models.mobilenet_v2(pretrained=True).features.to(device).eval()
def extract_features(img_tensor):
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(device)  # shape: (1, 3, 224, 224)
        features = cnn(img_tensor)  # shape: (1, 1280, 7, 7)
        pooled = F.adaptive_avg_pool2d(features, (1, 1))  # shape: (1, 1280, 1, 1)
        return pooled.view(-1).cpu().numpy()  # shape: (1280,)


def tsne():



    data = get_data()  #img, stats, label
    all_types = deencode_types()

    image_features = []
    labels = []

    for img_tensor, _, label in data:
        feat = extract_features(img_tensor)
        image_features.append(feat)

        type1_idx = label.argmax().item()
        type1 = all_types[type1_idx]
        labels.append(type1)

    image_features = np.stack(image_features)
    labels = np.array([t.title() for t in labels])

    pca = PCA(n_components=50)
    image_pca = pca.fit_transform(image_features)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_emb = tsne.fit_transform(image_pca)

    #thank you chatgpt for making my dreams reality
    type_colors = {
        'Normal': "#A8A77A",
        'Fire': "#EE8130",
        'Water': "#6390F0",
        'Electric': "#F7D02C",
        'Grass': "#7AC74C",
        'Ice': "#96D9D6",
        'Fighting': "#C22E28",
        'Poison': "#A33EA1",
        'Ground': "#E2BF65",
        'Flying': "#A98FF3",
        'Psychic': "#F95587",
        'Bug': "#A6B91A",
        'Rock': "#B6A136",
        'Ghost': "#735797",
        'Dragon': "#6F35FC",
        'Dark': "#705746",
        'Steel': "#B7B7CE",
        'Fairy': "#D685AD",
    }

    colors = [type_colors.get(t, "#000000") for t in labels]


    plt.figure(figsize=(10, 8))
    for t in sorted(set(labels)):
        idxs = labels == t
        plt.scatter(tsne_emb[idxs, 0], tsne_emb[idxs, 1], label=t, c=type_colors[t], s=10)

    plt.title("t-SNE on Pokémon Stats")
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig("tsne_stats_by_type.png", dpi=300)


tsne()


from collections import defaultdict

def plot_type_distribution():
    df = create_or_load_dataframe()
    type_counts = defaultdict(int)

    for _, row in df.iterrows():
        t1 = row["type1"]
        t2 = row["type2"]
        
        if pd.notna(t2):
            #if th epokemon has 2 types, add for both ? idk how we want to count it otherwise
            type_counts[t1] += 1
            type_counts[t2] += 1
        else:
            type_counts[t1] += 1


    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    labels, counts = zip(*sorted_types)

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.title("Number of Instances per Pokémon Type")
    plt.xlabel("Type")
    plt.ylabel("Instance Count (Split for Dual-Types)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("type_distribution.png")


plot_type_distribution()



"""
so the new code did this to the tabular data :🥲

Early stopping triggered.
Saved tabular model to pokemon_model_tabular.pt
Accuracy:  0.1178
Precision: 0.3854
Recall:    0.2240
F1-score:  0.2662
AUC-ROC:   0.5954
Using MPS
"""

"""
and did this to the image thing: 

Early stopping triggered.
Per-label accuracy: 0.9165
Accuracy:  0.1778
Precision: 0.4408
Recall:    0.3053
F1-score:  0.3454
"""


