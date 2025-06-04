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

    #normalize names just in case
    df['name'] = df['name'].apply(normalize)
    #df['name'] = df['name'].str.lower()
    #.str.replace(" ", "").str.replace("-", "")

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
            #fairly certains theres only pngs
            if fname.lower().endswith((".png")):
                full_image_path = os.path.join(pokemon_folder, fname)

                dataset.append({
                    "image_path": full_image_path,
                    "type1": row["type1"],
                    "type2": row["type2"],
                    #copy everything except the non numerical stuff
                    # TODO was the dropping of non numerical stuff for now, and we need to one-hot encode, or was it for a reason?
                    "stats": {k: clean_stat(v) for k, v in row.drop(["name", "type1", "type2", "japanese_name", "classfication", "abilities"]).items()}
                })

    create_overlap_analysis_csv(df, image_root)

    combined_df = pd.DataFrame(dataset)
    combined_df.to_pickle(pickle_path)  #pickleee
    return combined_df

# TODO should look into whether white/black background needs to be
def image_preprocess():
    return transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet req idk will we use that? otherwise change this
])

def image_unpreprocess(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return tensor * std + mean


#got fucking damnit the stats were strings :() so it set them all to 0 without complaining...... types in python fun
def tab_preprocess(df):
    combined_df = df
    stats_matrix = np.stack([[v for v in s.values()] for s in combined_df['stats']])

    print("Unique stat rows:", np.unique(stats_matrix, axis=0).shape[0])
    print(stats_matrix[0])
    #chatgpt told me this is a thing
    scaler = StandardScaler()
    scaled_stats = scaler.fit_transform(stats_matrix)
    combined_df["scaled_stats"] = [row for row in scaled_stats]
    print(scaled_stats[0])
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
def sample(df, idx, all_types, tfm):
    row = df.iloc[idx]

    image = Image.open(row["image_path"]).convert("RGB")
    image = tfm(image)

    tab = torch.tensor(row["scaled_stats"], dtype=torch.float32)
    label = encode_types(row["type1"], row["type2"], all_types)

    return image, tab, label

"""
Main call function
returns data in form of (image_tensor, stats_tensor, label_tensor) list (for now)
The types are kinda stupid, asked chatgpt how to do it, got like one hot vector thing but now you need deencode types to like get the types so maybe we need to change that or make an easier map
"""

#no idea what format do we want as output?
def get_data():
    df = create_or_load_dataframe()
    df = tab_preprocess(df)

    #make type list
    all_types = sorted(set(df['type1']) | set(df['type2'].dropna()))

    data = []
    for idx in range(len(df)):
        image, stats, label = sample(df, idx, all_types, image_preprocess())
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

def get_dataset():
    data = get_data()
    dataset = Pokemon(data)

    train_size = int(0.8 * len(dataset))
    train_ds, test_ds = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)
    
    return train_loader, test_loader

