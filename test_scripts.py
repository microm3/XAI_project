import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import random
from data import get_data, image_unpreprocess, deencode_types, add_white_background

DO_SINGLE_IMAGE_ORIGINAL_TEST = True
DO_IMAGE_WITH_STATS_TEST = True

def test(idx = 0):
    data = get_data()
    
    for image, stats, label in data:
        print(image.shape) 
        print(stats.shape)  
        print(label.shape)  
        break

    image, stats, label = data[idx]
    types = deencode_types()
    image = image_unpreprocess(image)
    #test image, YES I LEARNED FROM MY MISTAKES
    plt.imshow(F.to_pil_image(image))
    plt.axis('off')
    
    plt.title(f"Types: {', '.join([t for i, t in enumerate(types) if label[i] == 1])}")
    plt.savefig("test.png")

    #test stat
    print("Scaled stats (first 10):", np.round(stats[:10].numpy(), 3))
    

# print n pokemons: their images, preprocessed images, and stats in one figure
def print_images_with_stats(number_of_images = 10):
    data = get_data()
    types = deencode_types()
    
    os.makedirs("summary_images", exist_ok=True)
    
    # remove existing files first. 
    for file in os.listdir("summary_images"):
        file_path = os.path.join("summary_images", file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    df_sample = pd.read_csv('pokemon.csv')
    all_stat_names = df_sample.drop(['name', 'type1', 'type2', 'japanese_name', 'classfication', 'abilities'], axis=1).columns.tolist()
    
    for i in range(number_of_images):
        index = random.randrange(0, len(data))
        image, stats, label = data[index]
    
        df_combined = pd.read_pickle("combined_dataset.pkl")
        image_path = df_combined.iloc[index]['image_path']
        original_image = add_white_background(Image.open(image_path))
    
        # processed_image_display = image_unpreprocess(image)
        processed_image_display = image

        
        fig = plt.figure(figsize=(24, 12))
        
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(original_image)
        ax1.axis('off')
        ax1.set_title(f"Original Image", fontsize=14)
        
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(F.to_pil_image(processed_image_display))
        ax2.axis('off')
        ax2.set_title(f"Processed Image", fontsize=14)
        
        pokemon_name = image_path.split('/')[1]
        fig.suptitle(f"Pokemon: {pokemon_name.title()} (Dataset Index: {index})", fontsize=16, y=0.95)
        
        ax3 = plt.subplot(2, 1, 2)
        y_pos = range(len(all_stat_names))
        bars = ax3.barh(y_pos, stats.numpy())
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(all_stat_names, fontsize=8)
        ax3.set_xlabel("Scaled Value", fontsize=12)
        ax3.set_title("All Features (Scaled)", fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        feature_colors = []
        for name in all_stat_names:
            if name.startswith('against_'):
                feature_colors.append('lightcoral')
            elif name in ['attack', 'defense', 'hp', 'sp_attack', 'sp_defense', 'speed']:
                feature_colors.append('lightblue')
            elif name in ['height_m', 'weight_kg']:
                feature_colors.append('lightgreen')
            else:
                feature_colors.append('lightgray')
        
        for bar, color in zip(bars, feature_colors):
            bar.set_color(color)
        
        types_text = f"Types: {', '.join([t for j, t in enumerate(types) if label[j] == 1])}"
        plt.figtext(0.5, 0.02, types_text, ha='center', fontsize=12)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightcoral', label='Type Effectiveness'),
            Patch(facecolor='lightblue', label='Battle Stats'),
            Patch(facecolor='lightgreen', label='Physical'),
            Patch(facecolor='lightgray', label='Other')
        ]
        ax3.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(f"summary_images/{pokemon_name}_{i+1}_index_{index}.png", dpi=150, bbox_inches='tight')
        plt.close()



if DO_IMAGE_WITH_STATS_TEST:    
    print_images_with_stats()

if DO_SINGLE_IMAGE_ORIGINAL_TEST:
    test()