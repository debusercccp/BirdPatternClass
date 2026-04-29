"""
train.py – Training end-to-end del Transformer autoencoder.

Uso:
    python train.py --data_dir ./dataset --epochs 60 --save bird_model --plot

Struttura dataset:
    dataset/
      specie_A/   *.wav
      specie_B/   *.wav
      ...

Cartella piatta (nessuna sottocartella) = modalita' autoencoder pura.
"""

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from preprocessing import build_preprocessing_pipeline, WINDOW_SIZE, N_MFCC
from model import build_autoencoder, train_autoencoder, save_models, detect_repeated_patterns


AUDIO_EXTS = ["*.wav", "*.mp3", "*.ogg", "*.flac"]


def collect_files(data_dir: str) -> tuple[list[str], list[int], list[str]]:
    data_dir = Path(data_dir)
    species  = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if not species:
        paths = []
        for ext in AUDIO_EXTS:
            paths.extend(glob.glob(str(data_dir / ext)))
        return paths, [0] * len(paths), ["unknown"]

    paths, labels, names = [], [], []
    for idx, sp in enumerate(species):
        found = []
        for ext in AUDIO_EXTS:
            found.extend(glob.glob(str(sp / ext)))
        paths  += found
        labels += [idx] * len(found)
        names.append(sp.name)
        print(f"  [{idx}] {sp.name:30s}: {len(found)} file")

    return paths, labels, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  required=True)
    parser.add_argument("--epochs",    type=int,   default=60)
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--d_model",   type=int,   default=128)
    parser.add_argument("--num_heads", type=int,   default=4)
    parser.add_argument("--num_layers",type=int,   default=3)
    parser.add_argument("--save",      default="bird_model")
    parser.add_argument("--plot",      action="store_true")
    args = parser.parse_args()

    # 1) File
    print(f"\n[1/5] Raccolta file da: {args.data_dir}")
    paths, labels, species_names = collect_files(args.data_dir)
    print(f"      Totale: {len(paths)} file, {len(species_names)} specie")
    if not paths:
        raise RuntimeError("Nessun file audio trovato.")

    # 2) Preprocessing
    print("\n[2/5] Preprocessing + feature extraction...")
    pipeline = build_preprocessing_pipeline()
    X = pipeline.fit_transform(paths)
    print(f"      Shape: {X.shape}")

    pipeline_path = f"{args.save}_pipeline.pkl"
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"      Pipeline salvata: {pipeline_path}")

    # 3) Split
    print("\n[3/5] Split train/val...")
    X_train, X_val = train_test_split(X, test_size=args.val_split, random_state=42)
    print(f"      Train: {X_train.shape[0]}  Val: {X_val.shape[0]}")

    n_features = X.shape[2]
    n_classes  = len(species_names) if len(species_names) > 1 else None

    # 4) Modello
    print("\n[4/5] Build Transformer autoencoder...")
    result = build_autoencoder(
        window_size=WINDOW_SIZE,
        n_features=n_features,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        n_classes=n_classes,
    )
    autoencoder, encoder = result[0], result[1]
    autoencoder.summary()

    print("\n      Training...")
    history = train_autoencoder(
        autoencoder, X_train, X_val,
        epochs=args.epochs,
        batch_size=args.batch,
    )

    # 5) Salvataggio
    print("\n[5/5] Salvataggio...")
    save_models(autoencoder, encoder, prefix=args.save)

    errors, is_pattern, thr = detect_repeated_patterns(autoencoder, X)
    print(f"\nAnalisi pattern su tutto il dataset:")
    print(f"  Finestre totali : {len(errors)}")
    print(f"  Pattern ripetuti: {is_pattern.sum()} ({is_pattern.mean()*100:.1f}%)")
    print(f"  Soglia errore   : {thr:.5f}")
    print(f"  Media errore    : {errors.mean():.5f}")

    if args.plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history.history["loss"],     label="train")
        axes[0].plot(history.history["val_loss"], label="val")
        axes[0].set_title("Reconstruction Loss (MSE)")
        axes[0].set_xlabel("Epoca")
        axes[0].legend()

        axes[1].hist(errors, bins=50, color="#4a90d9", edgecolor="white")
        axes[1].axvline(thr, color="crimson", linestyle="--", label=f"soglia={thr:.4f}")
        axes[1].set_title("Distribuzione Reconstruction Error")
        axes[1].set_xlabel("MSE")
        axes[1].legend()

        plt.tight_layout()
        plot_path = f"{args.save}_analysis.png"
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot salvato: {plot_path}")
        plt.show()

    print(f"\nCompletato. Avvia predizione realtime con:")
    print(f"  python realtime.py --model {args.save} --threshold {thr:.5f}")


if __name__ == "__main__":
    main()
