# # run_rnn.py
# """
# How to use
# -----
# From the assignment1/ directory:
#     python -m experiments.run_rnn --config config/rnn.yaml

# Override the model type/tokenization from the CLI without editing the yaml:
#     python -m experiments.run_rnn --config config/rnn.yaml --cell-type gru --mode patch
# """

# import argparse
# import json
# import time
# from pathlib import Path

# import torch
# import torch.nn as nn
# import yaml

# from src.data.dataloader import create_dataloaders
# from src.data.pipeline import create_normalized_datasets
# from src.models.rnn import RecurrentClassifier
# from src.utils.seed import set_seed


# def load_config(path: str) -> dict:
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


# def parse_args():
#     parser = argparse.ArgumentParser(description="Train the RNN (LSTM/GRU) classifier.")
#     parser.add_argument("--config", type=str, default="config/rnn.yaml")
#     parser.add_argument("--cell-type", type=str, default=None, choices=["lstm", "gru"])
#     parser.add_argument("--mode", type=str, default=None, choices=["row", "col", "patch"])
#     parser.add_argument("--epochs", type=int, default=None)
#     parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
#     return parser.parse_args()


# # TRAIN / VALIDATE LOOPS
# def run_epoch(model, loader, criterion, device, optimizer=None):
#     is_train = optimizer is not None
#     model.train() if is_train else model.eval()

#     total_loss = 0.0
#     correct = 0
#     total = 0

#     context = torch.enable_grad() if is_train else torch.no_grad()

#     with context:
#         for images, labels in loader:
#             images = images.to(device)
#             labels = labels.to(device)

#             if is_train:
#                 optimizer.zero_grad()

#             logits = model(images)
#             loss = criterion(logits, labels)

#             if is_train:
#                 loss.backward()
#                 optimizer.step()

#             total_loss += loss.item() * images.size(0)
#             predictions = logits.argmax(dim=1)
#             correct += (predictions == labels).sum().item()
#             total += labels.size(0)

#     epoch_loss = total_loss / total
#     epoch_acc = correct / total

#     return epoch_loss, epoch_acc


# def train_model(model, train_loader, val_loader, epochs, learning_rate, weight_decay, device, checkpoint_path):
#     model = model.to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

#     history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
#     best_val_acc = 0.0

#     for epoch in range(epochs):
#         start = time.time()

#         train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
#         val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)

#         history["train_loss"].append(train_loss)
#         history["train_acc"].append(train_acc)
#         history["val_loss"].append(val_loss)
#         history["val_acc"].append(val_acc)

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), checkpoint_path)

#         elapsed = time.time() - start
#         print(
#             f"Epoch [{epoch + 1}/{epochs}] "
#             f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | "
#             f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} "
#             f"({elapsed:.1f}s)"
#         )

#     print(f"Best validation accuracy: {best_val_acc:.4f} (checkpoint: {checkpoint_path})")

#     return history, best_val_acc

# def main():
#     args = parse_args()
#     config = load_config(args.config)

#     if args.cell_type is not None:
#         config["model"]["cell_type"] = args.cell_type
#     if args.mode is not None:
#         config["model"]["mode"] = args.mode
#     if args.epochs is not None:
#         config["training"]["epochs"] = args.epochs

#     set_seed(config["seed"])

#     device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     data_cfg = config["data"]

#     train_subset, val_subset, test_dataset, mean, std = create_normalized_datasets(
#         data_dir=data_cfg["data_dir"],
#         validation_size=data_cfg["validation_size"],
#         seed=config["seed"],
#     )
#     print(f"Train mean/std computed on train split only: mean={mean:.4f}, std={std:.4f}")
#     print(f"Train size: {len(train_subset)} | Val size: {len(val_subset)} | Test size: {len(test_dataset)}")

#     train_loader, val_loader, test_loader = create_dataloaders(
#         train_subset,
#         val_subset,
#         test_dataset,
#         batch_size=data_cfg["batch_size"],
#         num_workers=data_cfg["num_workers"],
#     )

#     # ---- Model ----
#     model_cfg = config["model"]
#     model = RecurrentClassifier(
#         img_size=data_cfg["image_size"],
#         num_classes=data_cfg["num_classes"],
#         cell_type=model_cfg["cell_type"],
#         mode=model_cfg["mode"],
#         patch_size=model_cfg["patch_size"],
#         hidden_size=model_cfg["hidden_size"],
#         num_layers=model_cfg["num_layers"],
#         bidirectional=model_cfg["bidirectional"],
#         dropout=model_cfg["dropout"],
#     )
#     print(
#         f"Model: RecurrentClassifier(cell_type={model_cfg['cell_type']}, "
#         f"mode={model_cfg['mode']}, hidden_size={model_cfg['hidden_size']})"
#     )

#     # ---- Checkpointing ----
#     train_cfg = config["training"]
#     checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "experiments/checkpoints"))
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
#     checkpoint_path = checkpoint_dir / f"best_rnn_{model_cfg['cell_type']}_{model_cfg['mode']}.pt"

#     # ---- Train ----
#     history, best_val_acc = train_model(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         epochs=train_cfg["epochs"],
#         learning_rate=train_cfg["learning_rate"],
#         weight_decay=train_cfg.get("weight_decay", 0.0),
#         device=device,
#         checkpoint_path=checkpoint_path,
#     )

#     # Final, single-shot test evaluation using the BEST checkpoint
#     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#     criterion = nn.CrossEntropyLoss()
#     test_loss, test_acc = run_epoch(model, test_loader, criterion, device, optimizer=None)
#     print(f"Final Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc:.4f}")

#     # Save history + summary next to the checkpoint
#     results_path = checkpoint_dir / f"history_rnn_{model_cfg['cell_type']}_{model_cfg['mode']}.json"
#     with open(results_path, "w") as f:
#         json.dump(
#             {
#                 "config": config,
#                 "history": history,
#                 "best_val_acc": best_val_acc,
#                 "test_loss": test_loss,
#                 "test_acc": test_acc,
#             },
#             f,
#             indent=2,
#         )
#     print(f"Saved training history to: {results_path}")


# if __name__ == "__main__":
#     main()