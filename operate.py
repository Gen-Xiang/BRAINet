import random

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import torch

from dataset.dataset import save_json
from models.BRAINet import BRAINet


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


@torch.no_grad()
def predict(model, x, apen, batch_size=64, device="cpu"):
    model.eval()
    probabilities = []
    for start in range(0, len(x), batch_size):
        bx = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
        ba = torch.as_tensor(apen[start:start + batch_size], dtype=torch.float32, device=device)
        probabilities.append(model(bx, ba).softmax(1).cpu().numpy())
    return np.concatenate(probabilities)


def train_fold(x, y, apen, train_idx, test_idx, config, folder, device):
    seed_everything(config["seed"])
    model = BRAINet().to(device)
    dataset = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]),
                            torch.from_numpy(apen[train_idx]))
    # A dedicated generator keeps training order independent of test iteration.
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(config["seed"]), num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total = 0.0
        for bx, by, ba in loader:
            bx, by, ba = bx.to(device), by.to(device), ba.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx, ba), by)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(by)
        history.append({"epoch": epoch, "train_loss": total / len(dataset)})
        print(f"{folder.name} epoch {epoch}/{config['epochs']} train_loss={total / len(dataset):.6f}", flush=True)
    pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
    # Save only the fixed final checkpoint. No test-based model selection.
    torch.save({"state_dict": model.state_dict(), "config": config, "epoch": config["epochs"]}, folder / "final.pt")
    save_json(folder / "model_size.json", {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    return predict(model, x[test_idx], apen[test_idx], config["batch_size"], device)
