# # run_transformer.py
# """
# how to use
# -----
# From the assignment1/ directory:

#     python -m experiments.run_transformer --config config/transformer.yaml

# Override hyperparameters from the CLI without editing the yaml:

#     python -m experiments.run_transformer --config config/transformer.yaml --patch-size 7 --epochs 10
# """

# import argparse
# import json
# import math
# import time
# from pathlib import Path

# import torch
# import torch.nn as nn
# import yaml

# from src.data.dataloader import create_dataloaders
# from src.data.pipeline import create_normalized_datasets
# from src.models.transformer import VisionTransformerClassifier
# from src.utils.seed import set_seed


# def load_config(path: str) -> dict:
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


# def parse_args():
#     parser = argparse.ArgumentParser(description="Train the Vision Transformer classifier.")
#     parser.add_argument("--config", type=str, default="config/transformer.yaml")
#     parser.add_argument("--patch-size", type=int, default=None)
#     parser.add_argument("--epochs", type=int, default=None)
#     parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
#     return parser.parse_args()


# # ---------------------------------------------------------------------------
# # LR SCHEDULE: linear warmup -> cosine decay
# # ---------------------------------------------------------------------------
# def build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
#     def lr_lambda(epoch):
#         if warmup_epochs > 0 and epoch < warmup_epochs:
#             return (epoch + 1) / warmup_epochs
#         progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
#         return 0.5 * (1.0 + math.cos(math.pi * progress))

#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# # ---------------------------------------------------------------------------
# # TRAIN / VALIDATE LOOPS
# # ---------------------------------------------------------------------------
# def run_epoch(model, loader, criterion, device, optimizer=None):
#     """
#     Shared loop for both training and evaluation.
#     If `optimizer` is provided, the model is put in train mode and weights
#     are updated; otherwise it runs in eval mode with no_grad.
#     """
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


# def train_model(
#     model,
#     train_loader,
#     val_loader,
#     epochs,
#     learning_rate,
#     weight_decay,
#     warmup_epochs,
#     device,
#     checkpoint_path,
# ):
#     model = model.to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     scheduler = build_warmup_cosine_scheduler(optimizer, warmup_epochs, epochs)

#     history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
#     best_val_acc = 0.0

#     for epoch in range(epochs):
#         start = time.time()

#         train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
#         val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)

#         current_lr = optimizer.param_groups[0]["lr"]
#         scheduler.step()

#         history["train_loss"].append(train_loss)
#         history["train_acc"].append(train_acc)
#         history["val_loss"].append(val_loss)
#         history["val_acc"].append(val_acc)
#         history["lr"].append(current_lr)

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), checkpoint_path)

#         elapsed = time.time() - start
#         print(
#             f"Epoch [{epoch + 1}/{epochs}] "
#             f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | "
#             f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} | "
#             f"LR: {current_lr:.6f} ({elapsed:.1f}s)"
#         )

#     print(f"Best validation accuracy: {best_val_acc:.4f} (checkpoint: {checkpoint_path})")

#     return history, best_val_acc



# def main():
#     args = parse_args()
#     config = load_config(args.config)

#     # CLI overrides (optional, do not require editing the yaml)
#     if args.patch_size is not None:
#         config["model"]["patch_size"] = args.patch_size
#     if args.epochs is not None:
#         config["training"]["epochs"] = args.epochs

#     set_seed(config["seed"])

#     device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # ---- Data: reuse the existing, tested pipeline ----
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
#     model = VisionTransformerClassifier(
#         img_size=data_cfg["image_size"],
#         in_channels=data_cfg["num_channels"],
#         num_classes=data_cfg["num_classes"],
#         patch_size=model_cfg["patch_size"],
#         embed_dim=model_cfg["embed_dim"],
#         num_heads=model_cfg["num_heads"],
#         num_layers=model_cfg["num_layers"],
#         mlp_ratio=model_cfg["mlp_ratio"],
#         dropout=model_cfg["dropout"],
#     )
#     print(
#         f"Model: VisionTransformerClassifier(patch_size={model_cfg['patch_size']}, "
#         f"embed_dim={model_cfg['embed_dim']}, num_heads={model_cfg['num_heads']}, "
#         f"num_layers={model_cfg['num_layers']})"
#     )

#     # ---- Checkpointing ----
#     train_cfg = config["training"]
#     checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "experiments/checkpoints"))
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
#     checkpoint_path = checkpoint_dir / f"best_transformer_p{model_cfg['patch_size']}.pt"

#     # ---- Train ----
#     history, best_val_acc = train_model(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         epochs=train_cfg["epochs"],
#         learning_rate=train_cfg["learning_rate"],
#         weight_decay=train_cfg.get("weight_decay", 0.0),
#         warmup_epochs=train_cfg.get("warmup_epochs", 0),
#         device=device,
#         checkpoint_path=checkpoint_path,
#     )

#     # ---- Final, single-shot test evaluation using the BEST checkpoint ----
#     # The test set is never used for model/checkpoint selection above, only here.
#     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#     criterion = nn.CrossEntropyLoss()
#     test_loss, test_acc = run_epoch(model, test_loader, criterion, device, optimizer=None)
#     print(f"Final Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc:.4f}")

#     # ---- Save history + summary next to the checkpoint ----
#     results_path = checkpoint_dir / f"history_transformer_p{model_cfg['patch_size']}.json"
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