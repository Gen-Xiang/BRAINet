from pathlib import Path
import argparse
import json
import os
import platform

import numpy as np
import pandas as pd
import torch

from dataset.dataset import load_dataset, epoch_zscore, make_splits, validate_splits, save_json, sha256
from operate import train_fold
from utils.features import approximate_entropy
from utils.metrics import PROBS, aggregate, summarize


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def run(args):
    config = json.loads(Path(args.config).read_text())
    if args.protocol:
        config["protocol"] = args.protocol
    if args.smoke:
        config["epochs"] = 1
    elif config["epochs"] != 25 or config["batch_size"] != 64 or config["learning_rate"] != 1e-4 or config["weight_decay"] != 1e-4:
        raise ValueError("Paper protocol requires 25 epochs, batch 64, Adam lr=wd=1e-4")
    if config["protocol"] not in {"fivefold", "loso"}:
        raise ValueError("Unknown protocol")
    folder = Path(args.output)
    if folder.exists():
        raise FileExistsError("Choose a new output directory to avoid mixing experiments")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    x, y, ids, apen = load_dataset(args.data)
    x = epoch_zscore(x)
    if apen is None:
        apen = approximate_entropy(x, args.jobs)
    apen = apen.astype(np.float32)
    folds = json.loads(Path(args.splits).read_text())["folds"] if args.splits else make_splits(y, ids, config["seed"], config["protocol"])
    validate_splits(folds, y, ids, config["protocol"])
    folder.mkdir(parents=True)
    save_json(folder / "splits.json", {"folds": folds, "protocol": config["protocol"], "seed": config["seed"],
                                      "source": "supplied" if args.splits else "generated_patient_stratified"})
    save_json(folder / "run.json", {**config, "smoke_only": args.smoke, "data_sha256": sha256(args.data),
                                    "split_sha256": sha256(folder / "splits.json"), "python": platform.python_version(),
                                    "torch": torch.__version__, "numpy": np.__version__,
                                    "normalization": "per-epoch per-channel temporal zscore",
                                    "checkpoint_selection": "fixed final epoch", "test_evaluation": "after training only"})
    np.save(folder / "apen.npy", apen)
    frames = []
    for fold in folds:
        out = folder / f"fold_{fold['fold']}"
        out.mkdir()
        tr = np.flatnonzero(np.isin(ids, fold["train"]))
        te = np.flatnonzero(np.isin(ids, fold["test"]))
        probs = train_fold(x, y, apen, tr, te, config, out, args.device)
        frame = pd.DataFrame({"epoch_index": te, "subject_id": ids[te], "true_label": y[te], "fold": fold["fold"]})
        frame[PROBS] = probs
        frame["pred_label"] = probs.argmax(1)
        frame.to_csv(out / "epoch_predictions.csv", index=False)
        frames.append(frame)
    oof = pd.concat(frames, ignore_index=True)
    if set(oof.epoch_index) != set(range(len(x))) or len(oof) != len(x):
        raise RuntimeError("Incomplete OOF coverage")
    oof.to_csv(folder / "oof_epoch_predictions.csv", index=False)
    aggregate(oof).to_csv(folder / "oof_subject_predictions.csv", index=False)
    save_json(folder / "metrics.json", summarize(oof, config["protocol"]))
    print(f"Completed. Metrics: {folder / 'metrics.json'}")


def main():
    p = argparse.ArgumentParser(description='Train BRAINet with patient-wise cross-validation')
    p.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--splits", help="Frozen split JSON; use the same file for all repeated analyses")
    p.add_argument("--protocol", choices=["fivefold", "loso"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--jobs", type=int, default=1, help="ApEn parallel workers")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--smoke", action="store_true", help="One epoch per fold; NOT a paper experiment")
    run(p.parse_args())


if __name__ == "__main__":
    main()
