# Pokémon Type Classification - XAI Project

## Overview
This project implements explainable AI (XAI) techniques for Pokémon multimodal, multilabel Pokemon classification. 

## Project Structure (Central Files)

```
XAI_project/
├── data.py                                      # Preprocessing pipeline
├── train_model.py                               # Multimodal model training
├── train_model_iso_img.py                      # Image-only model
├── train_model_iso_tab.py                      # Tabular-only model
├── XAI_global_shap_analysis.py                 # SHAP explanations
├── XAI_hue_shift_analysis.py                   # Color dependency analysis
├── XAI_global_grayscale_analysis.py            # Grayscale performance analysis
├── XAI_local_explanations.py                   # Local explanations (saliency, modality contribution)
├── XAI_global_colour_associations/             # Color associations analysis
├── continuity_testing_shap.py                  # SHAP continuity evaluation
├── continuity_testing_hue_shift.py             # Hue shift continuity evaluation
├── continuity_testing_colour_associations.py   # Color association continuity evaluation
├── dataset/                                    # Dataset
│   ├── pokemon.csv                             # Tabular Pokemon data
│   └── pokemon_sprites/                        # Pokemon image dataset
├── scrape_sprites.py                           # Image data collection script
└── combined_dataset.pkl                        # Preprocessed dataset cache
```

## Model Architecture
- **Image Branch**: MobileNet v2-based CNN for visual feature extraction
- **Tabular Branch**: Multi-layer perceptron for statistical features
- **Fusion**: Concatenated features with classification head
- **Output**: Multi-label classification for 18 Pokémon types

## XAI Methods Implemented

### Global Explanations
- **SHAP**: Feature importance for tabular data (`XAI_global_shap_analysis.py`)
- **Hue Shift Analysis**: Color dependency evaluation (`XAI_hue_shift_analysis.py`)
- **Grayscale Analysis**: Performance without color information (`XAI_global_grayscale_analysis.py`)
- **Color Associations**: Colour-type associations (`XAI_global_colour_associations/`)

### Local Explanations
- **Saliency Maps**: Pixel-level importance visualization (`XAI_local_explanations.py`)
- **SmoothGrad**: Noise-reduced gradient visualization (`XAI_local_explanations.py`)
- **Modality Contribution**: Image vs tabular contribution per prediction (`XAI_local_explanations.py`)

### Explanation Evaluation (CO-12 Continuity Criterion)
- **SHAP Continuity**: Tests robustness of feature importance rankings to tabular noise
- **Hue Shift Continuity**: Tests model robustness to image noise on hue-shifted inputs
- **Color Association Continuity**: Tests robustness of pure color-type associations to RGB noise

## Data Preprocessing Pipeline 
### Feature Selection
#### Excluded features
Out of the 41 original 24 features were excluded. The following features were excluded: "name", "japanese_name", "type1", "type2", "classfication", "abilities", "against_*"

#### Retained Features:
Basic stats (HP, Attack, Defense, Speed, etc.), physical attributes (Height, Weight), generation, Pokedex number, legendary status, breeding characteristics, and percentage male.

### Image Processing
- **Size**: 224x224 pixels 
- **Background**: Transparent backgrounds converted to white
- **Normalization**: ImageNet standards (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Dataset Merging of Tabular and Visual Data
Matched 776 Pokemon with both image and tabular data based on normalized Pokemon name (lowecasing, removing spaces, hyphens, underscores)