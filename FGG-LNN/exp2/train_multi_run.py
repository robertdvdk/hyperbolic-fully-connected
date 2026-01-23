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
import numpy as np
import random

# Your custom models
from mlr_layers import (
    Lorentz_fully_connected_forward,
    Lorentz_fully_connected_mlr_angle,
    Lorentz_fully_connected_mlr_dist,
    BdeirLorentzMLR_model,
    Baseline_Euclidean,
    Poincare_MLR_Shimizu_van_Spengler,
)

MODEL_REGISTRY = {
    "euclidean": Baseline_Euclidean,
    "lorentz_forward": Lorentz_fully_connected_forward,
    "lorentz_angle": Lorentz_fully_connected_mlr_angle,
    "lorentz_dist": Lorentz_fully_connected_mlr_dist,
    "bdeir": BdeirLorentzMLR_model,
    "poincare": Poincare_MLR_Shimizu_van_Spengler,
}


def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(
    features_path: str, dataset_name: str, batch_size: int
) -> tuple[list, DataLoader, DataLoader, int]:
    base_path = os.path.join(features_path, dataset_name)
    train_data_paths = []
    augmented_path = os.path.join(base_path, "augmented_train")
    if os.path.exists(augmented_path):
        feature_files = sorted(
            glob.glob(os.path.join(augmented_path, "epoch_*_features.pt"))
        )
        label_files = sorted(
            glob.glob(os.path.join(augmented_path, "epoch_*_labels.pt"))
        )
        train_data_paths = list(zip(feature_files, label_files))
    else:
        train_data_paths = [
            (
                os.path.join(base_path, "train_features.pt"),
                os.path.join(base_path, "train_labels.pt"),
            )
        ]
    val_features = torch.load(
        os.path.join(base_path, "valid_features.pt"), map_location="cpu"
    )
    val_labels = torch.load(
        os.path.join(base_path, "valid_labels.pt"), map_location="cpu"
    )
    test_features = torch.load(
        os.path.join(base_path, "test_features.pt"), map_location="cpu"
    )
    test_labels = torch.load(
        os.path.join(base_path, "test_labels.pt"), map_location="cpu"
    )
    val_dataset = TensorDataset(val_features, val_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )
    num_classes = len(torch.unique(val_labels))
    return train_data_paths, val_loader, test_loader, num_classes


def train_one_epoch(
    model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device
) -> tuple[float, float]:
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


def evaluate(
    model: nn.Module, loader: DataLoader, criterion, device: torch.device
) -> dict:
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
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    return {
        "loss": total_loss / total_samples,
        "top1_accuracy": (correct_top1 / total_samples) * 100,
        "top5_accuracy": (correct_top5 / total_samples) * 100,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def log_results_to_file(filepath: str, header: str, data_row: str):
    """Appends a results row to the specified file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    write_header = not os.path.exists(filepath)
    with open(filepath, "a") as f:
        if write_header:
            f.write(header + "\n")
        f.write(data_row + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train a classification head on pre-extracted features."
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of times to run the experiment.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Initial random seed.")
    parser.add_argument(
        "--features_path",
        type=str,
        required=True,
        help="Path to the directory of saved features.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["aircraft", "flowers", "cars", "dogs-custom", "cub"],
        help="Name of the dataset to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODEL_REGISTRY.keys(),
        help="Name of the model head to use.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="classification-head-experiments",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--curvature",
        type=float,
        default=0.1,
        help="Curvature `k` for hyperbolic models.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    all_best_epochs = []  # <-- MODIFIED: Renamed from all_stopping_epochs
    all_best_val_metrics = []
    all_best_test_metrics = []

    for i in range(args.num_runs):
        current_seed = args.seed + i
        print(
            f"\n\n{'=' * 20} RUN {i + 1}/{args.num_runs} (Seed: {current_seed}) {'=' * 20}"
        )
        set_seed(current_seed)

        run = wandb.init(project=args.wandb_project, config=args, reinit=True)
        wandb.run.name = f"{args.dataset}_{args.model}_c{args.curvature}_lr{args.lr}_seed{current_seed}"

        train_data_paths, val_loader, test_loader, num_classes = get_dataloaders(
            args.features_path, args.dataset, args.batch_size
        )

        model_class = MODEL_REGISTRY[args.model]
        model = model_class(
            in_features=2048, out_features=num_classes, k=args.curvature
        ).to(device)
        model = torch.compile(model)

        wandb.watch(model, log="all", log_freq=100)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        best_val_acc = 0.0
        epochs_no_improve = 0
        patience = 7

        best_epoch_num = 0  # <-- MODIFIED: To store the best epoch number for this run
        best_epoch_val_metrics = None
        best_epoch_test_metrics = None

        for epoch in range(args.epochs):
            current_epoch_num = epoch + 1
            print(f"\n--- Epoch {current_epoch_num}/{args.epochs} ---")

            feature_path, label_path = train_data_paths[epoch % len(train_data_paths)]
            train_features = torch.load(feature_path, map_location="cpu")
            train_labels = torch.load(label_path, map_location="cpu")
            train_dataset = TensorDataset(train_features, train_labels)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=0,
            )

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_metrics = evaluate(model, val_loader, criterion, device)

            # Evaluate on test set only if necessary to save time
            current_test_metrics = None
            if val_metrics["top1_accuracy"] > best_val_acc:
                current_test_metrics = evaluate(model, test_loader, criterion, device)

            log_dict = {
                "epoch": current_epoch_num,
                "train/loss": train_loss,
                "train/top1_accuracy": train_acc,
                "val/loss": val_metrics["loss"],
                "val/top1_accuracy": val_metrics["top1_accuracy"],
                "val/top5_accuracy": val_metrics["top5_accuracy"],
                "val/precision": val_metrics["precision"],
                "val/recall": val_metrics["recall"],
                "val/f1_score": val_metrics["f1_score"],
            }
            if current_test_metrics:
                log_dict.update(
                    {
                        "test/loss": current_test_metrics["loss"],
                        "test/top1_accuracy": current_test_metrics["top1_accuracy"],
                        "test/top5_accuracy": current_test_metrics["top5_accuracy"],
                        "test/precision": current_test_metrics["precision"],
                        "test/recall": current_test_metrics["recall"],
                        "test/f1_score": current_test_metrics["f1_score"],
                    }
                )
            wandb.log(log_dict)

            print(
                f"Epoch {current_epoch_num:02d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_metrics['top1_accuracy']:.2f}%"
            )

            if val_metrics["top1_accuracy"] > best_val_acc:
                best_val_acc = val_metrics["top1_accuracy"]
                epochs_no_improve = 0
                print(
                    f"✨ New best val acc: {best_val_acc:.2f}%. Storing metrics from epoch {current_epoch_num}."
                )

                best_epoch_num = (
                    current_epoch_num  # <-- MODIFIED: Record the best epoch number
                )
                best_epoch_val_metrics = val_metrics
                best_epoch_test_metrics = current_test_metrics
                wandb.summary["best_val_accuracy"] = best_val_acc
                wandb.summary["best_test_accuracy"] = best_epoch_test_metrics[
                    "top1_accuracy"
                ]
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"🛑 Early stopping triggered after {patience} epochs without improvement."
                    )
                    break

        all_best_epochs.append(
            best_epoch_num
        )  # <-- MODIFIED: Append the best epoch number

        # Handle cases where training finishes without ever improving
        if best_epoch_val_metrics is None:
            best_epoch_val_metrics = val_metrics
            best_epoch_test_metrics = evaluate(
                model, test_loader, criterion, device
            )  # Final test eval

        all_best_val_metrics.append(best_epoch_val_metrics)
        all_best_test_metrics.append(best_epoch_test_metrics)

        run.finish()

    print("\n--- All runs complete. Aggregating results. ---")

    mean_best_epoch = np.mean(all_best_epochs)  # <-- MODIFIED
    std_best_epoch = np.std(all_best_epochs)  # <-- MODIFIED

    def get_stats(metrics_list, key):
        values = [m[key] for m in metrics_list]
        return np.mean(values), np.std(values)

    val_acc_mean, val_acc_std = get_stats(all_best_val_metrics, "top1_accuracy")
    val_top5_acc_mean, val_top5_acc_std = get_stats(
        all_best_val_metrics, "top5_accuracy"
    )
    val_prec_mean, val_prec_std = get_stats(all_best_val_metrics, "precision")
    val_recall_mean, val_recall_std = get_stats(all_best_val_metrics, "recall")
    val_f1_mean, val_f1_std = get_stats(all_best_val_metrics, "f1_score")
    val_loss_mean, val_loss_std = get_stats(all_best_val_metrics, "loss")

    test_acc_mean, test_acc_std = get_stats(all_best_test_metrics, "top1_accuracy")
    test_top5_acc_mean, test_top5_acc_std = get_stats(
        all_best_test_metrics, "top5_accuracy"
    )
    test_prec_mean, test_prec_std = get_stats(all_best_test_metrics, "precision")
    test_recall_mean, test_recall_std = get_stats(all_best_test_metrics, "recall")
    test_f1_mean, test_f1_std = get_stats(all_best_test_metrics, "f1_score")
    test_loss_mean, test_loss_std = get_stats(all_best_test_metrics, "loss")

    # <-- MODIFIED: Header updated
    header = (
        "model,dataset,lr,curvature,batch_size,weight_decay,"
        "best_epochs,best_epoch_mean,best_epoch_std,"
        "val_acc_mean,val_acc_std,val_top5_acc_mean,val_top5_acc_std,"
        "val_precision_mean,val_precision_std,val_recall_mean,val_recall_std,"
        "val_f1_score_mean,val_f1_score_std,val_loss_mean,val_loss_std,"
        "test_acc_mean,test_acc_std,test_top5_acc_mean,test_top5_acc_std,"
        "test_precision_mean,test_precision_std,test_recall_mean,test_recall_std,"
        "test_f1_score_mean,test_f1_score_std,test_loss_mean,test_loss_std"
    )

    best_epochs_str = "-".join(map(str, all_best_epochs))  # <-- MODIFIED

    # <-- MODIFIED: Data row updated
    data_row = (
        f"{args.model},{args.dataset},{args.lr},{args.curvature},{args.batch_size},{args.weight_decay},"
        f'"{best_epochs_str}",{mean_best_epoch:.2f},{std_best_epoch:.2f},'
        f"{val_acc_mean:.4f},{val_acc_std:.4f},{val_top5_acc_mean:.4f},{val_top5_acc_std:.4f},"
        f"{val_prec_mean:.4f},{val_prec_std:.4f},{val_recall_mean:.4f},{val_recall_std:.4f},"
        f"{val_f1_mean:.4f},{val_f1_std:.4f},{val_loss_mean:.4f},{val_loss_std:.4f},"
        f"{test_acc_mean:.4f},{test_acc_std:.4f},{test_top5_acc_mean:.4f},{test_top5_acc_std:.4f},"
        f"{test_prec_mean:.4f},{test_prec_std:.4f},{test_recall_mean:.4f},{test_recall_std:.4f},"
        f"{test_f1_mean:.4f},{test_f1_std:.4f},{test_loss_mean:.4f},{test_loss_std:.4f}"
    )

    results_filepath = "exp2/results.csv"
    log_results_to_file(results_filepath, header, data_row)

    print(f"\n✅ Results successfully appended to {results_filepath}")


if __name__ == "__main__":
    main()
