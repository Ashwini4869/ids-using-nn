import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from model import Model
from dataloader import test_loader
from constants import TEST_MODEL_PATH


def load_model(model_path):
    model = Model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_preds = []
    all_labels = []

    print("Running inference on test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.float().to(device)
            outputs = model(inputs)
            predictions = (outputs >= 0.5).float().cpu().numpy()

            all_preds.extend(predictions)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def main():
    # Check if model exists
    model_path = TEST_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Load model
    print("Loading model...")
    model = load_model(model_path)

    # Run inference
    predictions, ground_truth = test_model(model, test_loader)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(ground_truth, predictions)

    # Print results
    print("\nTest Results:")
    print("-" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
