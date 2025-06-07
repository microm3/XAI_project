from train_model import train, show_predictions
import os


if __name__ == "__main__":
    if not os.path.exists('pokemon_model.pt'):
        print("No saved model found. Training first")
        train()
    else:
        print("Found existing model. Showing predictions")
        
    show_predictions()