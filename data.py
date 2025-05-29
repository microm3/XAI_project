#create dataset combine from both images + stats ---> get_data returns (image_tensor, stats_tensor, label_tensor)
import os
import pandas as pd
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def create_or_load_dataframe():
    pickle_path = "combined_dataset.pkl"
    if os.path.exists(pickle_path):
        return pd.read_pickle(pickle_path)

    csv_path = "pokemon.csv"
    image_path = "pokemon_sprites"
    df = pd.read_csv(csv_path)

    #normalize names just in case
    df['name'] = df['name'].str.lower().str.replace(" ", "").str.replace("-", "")

    dataset = []

    for _, row in df.iterrows():
        #join images with name on name
        pokemon_name = row['name']
        pokemon_folder = os.path.join(image_path, pokemon_name)
        
        #there are i think less stats then images but this was the biggest stats i could find
        if not os.path.isdir(pokemon_folder):
            print(f"missing {pokemon_name}")
            continue

        for fname in os.listdir(pokemon_folder):
            #fairly certains theres only pngs
            if fname.lower().endswith((".png")):
                image_path = os.path.join(pokemon_folder, fname)
                dataset.append({
                    "image_path": image_path,
                    "type1": row["type1"],
                    "type2": row["type2"],
                    #copy everything except the non numerical stuff
                    "stats": row.drop(["name", "type1", "type2", "japanese_name", "classfication", "abilities"]).to_dict()
                })


    combined_df = pd.DataFrame(dataset)
    combined_df.to_pickle(pickle_path)  #pickleee
    return combined_df

    
def image_preprocess():
    return transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet req idk will we use that? otherwise change this
                         
])
    
def tab_preprocess(df):
    combined_df = df
    stats_matrix = np.stack([list(s.values()) for s in combined_df['stats']])
    #chatgpt told me this is a thing
    scaler = StandardScaler()
    scaled_stats = scaler.fit_transform(stats_matrix)
    combined_df["scaled_stats"] = list(scaled_stats)
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

    return data  #(image_tensor, stats_tensor, label_tensor)


def test(idx = 0):
    data = get_data()
    
    for image, stats, label in data:
        print(image.shape) 
        print(stats.shape)  
        print(label.shape)  
        break

    image, stats, label = data[idx]
    types = deencode_types()
    #test image, YES I LEARNED FROM MY MISTAKES
    plt.imshow(F.to_pil_image(image))
    plt.axis('off')
    
    plt.title(f"Types: {', '.join([t for i, t in enumerate(types) if label[i] == 1])}")
    plt.savefig("test.png")

    #test stat
    print("stats:", stats[:10])
    
test()