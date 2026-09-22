from pathlib import Path
import argparse

import mne
import numpy as np
import pandas as pd

from dataset.dataset import epoch_zscore, save_json
from utils.channels import CHANNELS


def preprocess_raw(raw):
    raw = raw.copy().pick(CHANNELS).reorder_channels(CHANNELS).load_data()
    if not np.isclose(raw.info["sfreq"], 1000):
        raise ValueError("The manuscript pipeline expects 1000 Hz raw recordings")
    raw.notch_filter([50], method="fir", phase="zero", verbose=False)
    raw.filter(1, 40, method="fir", phase="zero", verbose=False)
    signal = raw.get_data()
    if not np.isfinite(signal).all():
        raise ValueError("Nonfinite raw signal")
    count = signal.shape[1] // 2000
    if count == 0:
        raise ValueError("Recording shorter than two seconds")
    # Reshape channels first, then transpose: never interleave channel/time axes.
    epochs = signal[:, :count * 2000].reshape(63, count, 2000).transpose(1, 0, 2)
    keep = np.max(np.abs(epochs), axis=(1, 2)) <= 180e-6
    epochs = epochs[keep]
    if not len(epochs):
        raise ValueError("All epochs rejected")
    epochs -= epochs.mean(axis=1, keepdims=True)
    epochs = mne.filter.resample(epochs, down=4, npad="auto", axis=-1, verbose=False)
    return epoch_zscore(epochs), {"initial_epochs": count, "retained_epochs": int(keep.sum()),
                                  "rejected_epochs": int((~keep).sum()), "retained_epoch_indices": np.flatnonzero(keep).tolist()}


def main():
    p = argparse.ArgumentParser(description="Preprocess raw EEG from a subject_id,label,path CSV")
    p.add_argument("manifest"); p.add_argument("output")
    a = p.parse_args()
    out = Path(a.output)
    if out.suffix != ".npz":
        raise ValueError("Output filename must end in .npz")
    if out.exists() or out.with_suffix(".preprocessing.json").exists():
        raise FileExistsError(out)
    frame = pd.read_csv(a.manifest, dtype={"subject_id": str})
    if not {"subject_id", "label", "path"} <= set(frame) or frame.empty:
        raise ValueError("Expected nonempty subject_id,label,path CSV")
    if frame.subject_id.isna().any() or not frame.label.isin([0, 1, 2]).all():
        raise ValueError("Invalid IDs or labels")
    if (frame.groupby("subject_id").label.nunique() != 1).any():
        raise ValueError("Conflicting labels")
    xs, ys, ids, audit = [], [], [], []
    for row in frame.itertuples():
        path = Path(row.path)
        if not path.is_absolute():
            path = Path(a.manifest).resolve().parent / path
        raw = mne.io.read_raw(path, preload=True, verbose=False)
        x, record = preprocess_raw(raw)
        xs.append(x); ys.extend([row.label] * len(x)); ids.extend([row.subject_id] * len(x))
        audit.append({"subject_id": row.subject_id, **record})
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=np.concatenate(xs), y=np.array(ys), subject_id=np.array(ids), channels=CHANNELS, sfreq=250)
    save_json(out.with_suffix(".preprocessing.json"), audit)


if __name__ == "__main__":
    main()
