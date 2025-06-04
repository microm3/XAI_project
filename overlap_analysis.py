# I want these numbers for the report :) 
import os
import pandas as pd

def create_overlap_analysis_csv(df, image_path):
    image_folders = [d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))]

    stats_names = set(df['name'].tolist())
    image_names = set(image_folders)
    overlapping_pokemon = stats_names.intersection(image_names)
    stats_only = stats_names - image_names
    images_only = image_names - stats_names
    analysis_data = []

    analysis_data.append({'category': 'SUMMARY', 'pokemon_name': 'total_stats_pokemon', 'count': len(df), 'description': 'Total Pokemon in stats file'})
    analysis_data.append({'category': 'SUMMARY', 'pokemon_name': 'total_image_pokemon', 'count': len(image_folders), 'description': 'Total unique Pokemon with images'})
    analysis_data.append({'category': 'SUMMARY', 'pokemon_name': 'overlapping_pokemon', 'count': len(overlapping_pokemon), 'description': 'Pokemon with both stats and images'})
    analysis_data.append({'category': 'SUMMARY', 'pokemon_name': 'coverage_percentage', 'count': round(100 * len(overlapping_pokemon) / len(df), 1), 'description': 'Percentage of stats Pokemon that have images'})

    for name in sorted(overlapping_pokemon):
        analysis_data.append({'category': 'OVERLAPPING', 'pokemon_name': name, 'count': 1, 'description': 'Has both stats and images'})

    for name in sorted(stats_only):
        analysis_data.append({'category': 'STATS_ONLY', 'pokemon_name': name, 'count': 1, 'description': 'Has stats but no images'})

    for name in sorted(images_only):
        analysis_data.append({'category': 'IMAGES_ONLY', 'pokemon_name': name, 'count': 1, 'description': 'Has images but no stats'})

    analysis_df = pd.DataFrame(analysis_data)
    analysis_df.to_csv("pokemon_overlap_analysis.csv", index=False)