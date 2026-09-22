from pathlib import Path
import argparse

from joblib import Parallel, delayed
import EntropyHub as EH
import numpy as np

from dataset.dataset import load_dataset, epoch_zscore
from utils.channels import CHANNELS


def channel_apen(signal):
    signal = np.asarray(signal, dtype=float)
    sd = signal.std()
    return 0.0 if sd == 0 else float(EH.ApEn(signal, m=2, r=0.2 * sd)[0][-1])


def approximate_entropy(x, jobs=1):
    values = Parallel(n_jobs=jobs)(delayed(channel_apen)(ch) for epoch in x for ch in epoch)
    result = np.asarray(values, dtype=np.float32).reshape(len(x), 63)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite approximate entropy")
    return result


def main():
    p = argparse.ArgumentParser(description="Compute reproducible ApEn features; preserve epoch order")
    p.add_argument("input"); p.add_argument("output"); p.add_argument("--jobs", type=int, default=1)
    a = p.parse_args()
    if Path(a.output).suffix != ".npz":
        raise ValueError("Output filename must end in .npz")
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    x, y, ids, _ = load_dataset(a.input)
    x = epoch_zscore(x)
    apen = approximate_entropy(x, a.jobs)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, x=x, y=y, subject_id=ids, apen=apen, channels=CHANNELS, sfreq=250)


if __name__ == "__main__":
    main()
