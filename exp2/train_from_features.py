import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
from sklearn.metrics import precision_recall_fscore_support
# Your custom models
from mlr_layers import (
    Lorentz_fully_connected_forward, Lorentz_fully_connected_mlr_angle,
    Lorentz_fully_connected_mlr_dist, BdeirLorentzMLR_model,
    Baseline_Euclidean, Poincare_MLR_Shimizu_van_Spengler
)

# MODIFICATION: Model registry for easy selection from CLI
MODEL_REGISTRY = {
    'euclidean': Baseline_Euclidean,
    'lorentz_forward': Lorentz_fully_connected_forward,
    'lorentz_angle': Lorentz_fully_connected_mlr_angle,
    'lorentz_dist': Lorentz_fully_connected_mlr_dist,
    'bdeir': BdeirLorentzMLR_model,
    'poincare': Poincare_MLR_Shimizu_van_Spengler,
}

def get_dataloaders(features_path: str, dataset_name: str, batch_size: int) -> tuple[list, DataLoader, DataLoader, int]:
    base_path = os.path.join(features_path, dataset_name)
    print(f"📁 Loading features from: {base_path}")
    train_data_paths = []
    augmented_path = os.path.join(base_path, "augmented_train")
    if os.path.exists(augmented_path):
        print("Found augmented training data. Will stream from files.")
        feature_files = sorted(glob.glob(os.path.join(augmented_path, "epoch_*_features.pt")))
        label_files = sorted(glob.glob(os.path.join(augmented_path, "epoch_*_labels.pt")))
        train_data_paths = list(zip(feature_files, label_files))
    else:
        print("Found single training feature file.")
        train_data_paths = [(
            os.path.join(base_path, "train_features.pt"),
            os.path.join(base_path, "train_labels.pt")
        )]
    val_features = torch.load(os.path.join(base_path, "valid_features.pt"), map_location='cpu')
    val_labels = torch.load(os.path.join(base_path, "valid_labels.pt"), map_location='cpu')
    test_features = torch.load(os.path.join(base_path, "test_features.pt"), map_location='cpu')
    test_labels = torch.load(os.path.join(base_path, "test_labels.pt"), map_location='cpu')
    val_dataset = TensorDataset(val_features, val_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)
    num_classes = len(torch.unique(val_labels))
    print(f"Found {num_classes} classes in the dataset.")
    return train_data_paths, val_loader, test_loader, num_classes

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss, correct_predictions, total_samples = 0.0, 0, 0
    for features, labels in tqdm(loader, desc="Training", leave=False):
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    return total_loss / total_samples, (correct_predictions / total_samples) * 100

def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> dict:
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    correct_top1, correct_top5, total_samples = 0, 0, 0
    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Evaluating", leave=False):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * features.size(0)
            _, top5_preds = torch.topk(outputs, 5, dim=1)
            total_samples += labels.size(0)
            correct_top1 += (top5_preds[:, 0] == labels).sum().item()
            correct_top5 += top5_preds.eq(labels.view(-1, 1)).any(dim=1).sum().item()
            all_preds.extend(top5_preds[:, 0].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    return {
        "loss": total_loss / total_samples, "top1_accuracy": (correct_top1 / total_samples) * 100,
        "top5_accuracy": (correct_top5 / total_samples) * 100, "precision": precision,
        "recall": recall, "f1_score": f1
    }

def main():
    parser = argparse.ArgumentParser(description="Train a classification head on pre-extracted features.")
    parser.add_argument('--features_path', type=str, required=True, help='Path to the directory of saved features.')
    parser.add_argument('--dataset', type=str, required=True, choices=['aircraft', 'flowers', 'cars', 'dogs-custom', 'cub'], help='Name of the dataset to use.')
    parser.add_argument('--model', type=str, required=True, choices=MODEL_REGISTRY.keys(), help='Name of the model head to use.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--wandb_project', type=str, default="classification-head-experiments", help="Weights & Biases project name.")
    parser.add_argument('--curvature', type=float, default=0.1, help='Curvature `k` for hyperbolic models.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer.')
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, config=args)
    wandb.run.name = f"{args.dataset}_{args.model}_c{args.curvature}_lr{args.lr}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    train_data_paths, val_loader, test_loader, num_classes = get_dataloaders(args.features_path, args.dataset, args.batch_size)
    
    print(f"🧠 Instantiating model: {args.model} with curvature k={args.curvature}")
    model_class = MODEL_REGISTRY[args.model]
    model = model_class(
        in_features=2048,
        out_features=num_classes,
        k=args.curvature
    ).to(device)

    # *** NEW: COMPILE THE MODEL ***
    print("Compiling model for performance...")
    # This uses Just-In-Time (JIT) compilation to optimize the model for faster execution.
    model = torch.compile(model)
    
    wandb.watch(model, log="all", log_freq=100)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_epochs = args.epochs
    
    # *** NEW: EARLY STOPPING SETUP ***
    best_val_acc = 0.0
    epochs_no_improve = 0
    patience = 7 # Number of epochs to wait for improvement before stopping

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        feature_path, label_path = train_data_paths[epoch % len(train_data_paths)]
        if len(train_data_paths) > 1:
            print(f"Loading training data from: {os.path.basename(feature_path)}")
        
        train_features = torch.load(feature_path, map_location='cpu')
        train_labels = torch.load(label_path, map_location='cpu')
        train_dataset = TensorDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        wandb.log({
            "epoch": epoch + 1, "train/loss": train_loss, "train/top1_accuracy": train_acc,
            "val/loss": val_metrics["loss"], "val/top1_accuracy": val_metrics["top1_accuracy"],
            "val/top5_accuracy": val_metrics["top5_accuracy"], "val/precision": val_metrics["precision"],
            "val/recall": val_metrics["recall"], "val/f1_score": val_metrics["f1_score"],
            "test/loss": test_metrics["loss"], "test/top1_accuracy": test_metrics["top1_accuracy"],
            "test/top5_accuracy": test_metrics["top5_accuracy"], "test/precision": test_metrics["precision"],
            "test/recall": test_metrics["recall"], "test/f1_score": test_metrics["f1_score"],
        })
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_metrics['top1_accuracy']:.2f}% | Test Acc: {test_metrics['top1_accuracy']:.2f}%")
        
        # *** NEW: EARLY STOPPING LOGIC ***
        if val_metrics["top1_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["top1_accuracy"]
            epochs_no_improve = 0  # Reset counter
            wandb.summary['best_val_accuracy'] = best_val_acc
            wandb.summary['best_test_accuracy'] = test_metrics["top1_accuracy"]
            print(f"✨ New best validation accuracy: {best_val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"Validation accuracy did not improve for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping triggered after {patience} epochs without improvement.")
                break # Exit the training loop
        
    wandb.finish()
    print("\n--- Training Complete ---")
    print("📈 Results have been logged to Weights & Biases.")

if __name__ == '__main__':
    main()