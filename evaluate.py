"""
evaluate.py – Clustering sullo spazio latente + curva Precision-Recall per specie.

Flusso:
  1. Estrae il vettore latente per ogni finestra tramite l'encoder.
  2. Riduce a 2D con UMAP (o PCA come fallback) per la visualizzazione.
  3. Trova il numero ottimale di cluster con silhouette score (K-Means).
  4. Associa ogni cluster alla specie piu' frequente al suo interno
     (majority vote sulle label di specie ground truth).
  5. Calcola precision, recall e F1 per ogni specie.
  6. Plotta: scatter latente colorato per cluster, curva Precision-Recall
     one-vs-rest per ogni specie, matrice di confusione cluster-specie.

Uso:
    python evaluate.py --data_dir ./dataset --model bird_model --plot
"""

import argparse
import glob
import pickle
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from model import load_models
from preprocessing import build_preprocessing_pipeline, WINDOW_SIZE

AUDIO_EXTS = ["*.wav", "*.mp3", "*.ogg", "*.flac"]


# ---- raccolta file con label ------------------------------------------------
def collect_files(data_dir: str) -> tuple[list[str], list[int], list[str]]:
    data_dir = Path(data_dir)
    species  = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not species:
        raise ValueError("evaluate.py richiede sottocartelle per specie.")

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


# ---- estrazione latente -----------------------------------------------------
def extract_latent(
    encoder,
    pipeline,
    paths: list[str],
    file_labels: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estrae vettori latenti per tutte le finestre.

    Returns:
        Z           : (N, d_model) vettori latenti
        win_labels  : (N,) label di specie per finestra
                      (ereditata dal file di origine)
    """
    print("\nEstrazione vettori latenti...")
    X = pipeline.transform(paths)   # (N, T, F)

    # origins_ mappa finestra -> (file_idx, frame_start)
    seq_builder = pipeline.named_steps["sequences"]
    origins     = seq_builder.origins_

    Z          = encoder.predict(X, verbose=0)          # (N, d_model)
    win_labels = np.array([file_labels[o[0]] for o in origins])

    print(f"  Finestre totali  : {len(Z)}")
    print(f"  Dimensione latente: {Z.shape[1]}")
    return Z, win_labels


# ---- silhouette per K ottimale ---------------------------------------------
def find_optimal_k(
    Z: np.ndarray,
    k_min: int = 2,
    k_max: int = 12,
) -> tuple[int, list[float]]:
    """
    Prova K da k_min a k_max e restituisce il K con silhouette massimo.
    """
    print(f"\nRicerca K ottimale (silhouette score, K={k_min}..{k_max})...")
    scores = []
    k_range = range(k_min, min(k_max + 1, len(Z)))

    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(Z)
        s   = silhouette_score(Z, lbl, sample_size=min(5000, len(Z)))
        scores.append(s)
        print(f"  K={k:2d}  silhouette={s:.4f}")

    best_k = list(k_range)[int(np.argmax(scores))]
    print(f"  K ottimale: {best_k}  (silhouette={max(scores):.4f})")
    return best_k, scores, list(k_range)


# ---- clustering e associazione specie ---------------------------------------
def cluster_and_assign(
    Z: np.ndarray,
    win_labels: np.ndarray,
    k: int,
    species_names: list[str],
) -> tuple[np.ndarray, dict, np.ndarray]:
    """
    Clusterizza con K-Means e associa ogni cluster alla specie
    piu' frequente (majority vote).

    Returns:
        cluster_ids    : (N,) indice cluster per finestra
        cluster_to_sp  : dict {cluster_id -> species_idx}
        pred_labels    : (N,) label di specie predetta per finestra
    """
    km          = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(Z)

    cluster_to_sp = {}
    for c in range(k):
        mask   = cluster_ids == c
        if mask.sum() == 0:
            cluster_to_sp[c] = 0
            continue
        counts = Counter(win_labels[mask].tolist())
        majority_sp = counts.most_common(1)[0][0]
        cluster_to_sp[c] = majority_sp
        sp_name = species_names[majority_sp]
        print(f"  Cluster {c:2d} -> {sp_name:30s}  ({counts})")

    pred_labels = np.array([cluster_to_sp[c] for c in cluster_ids])
    return cluster_ids, cluster_to_sp, pred_labels


# ---- riduzione dimensionale per plot ----------------------------------------
def reduce_2d(Z: np.ndarray) -> np.ndarray:
    if HAS_UMAP:
        print("\nRiduzione dimensionale con UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        return reducer.fit_transform(Z)
    else:
        print("\nUMAP non disponibile, uso PCA...")
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(Z)


# ---- plot completo ----------------------------------------------------------
def plot_evaluation(
    Z2d: np.ndarray,
    cluster_ids: np.ndarray,
    win_labels: np.ndarray,
    pred_labels: np.ndarray,
    species_names: list[str],
    k: int,
    k_range: list[int],
    sil_scores: list[float],
    save_path: str = "evaluation.png",
):
    n_species = len(species_names)
    colors    = plt.cm.tab10(np.linspace(0, 1, max(k, n_species)))

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ---- 1. Scatter latente colorato per cluster ----------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    for c in range(k):
        mask = cluster_ids == c
        ax1.scatter(Z2d[mask, 0], Z2d[mask, 1], s=4, alpha=0.4,
                    color=colors[c], label=f"Cluster {c}")
    ax1.set_title("Spazio latente – cluster")
    ax1.legend(markerscale=3, fontsize=7)
    ax1.set_xlabel("dim 1")
    ax1.set_ylabel("dim 2")

    # ---- 2. Scatter latente colorato per specie ground truth ----------------
    ax2 = fig.add_subplot(gs[0, 1])
    for s in range(n_species):
        mask = win_labels == s
        ax2.scatter(Z2d[mask, 0], Z2d[mask, 1], s=4, alpha=0.4,
                    color=colors[s], label=species_names[s])
    ax2.set_title("Spazio latente – specie (ground truth)")
    ax2.legend(markerscale=3, fontsize=7)
    ax2.set_xlabel("dim 1")
    ax2.set_ylabel("dim 2")

    # ---- 3. Silhouette score vs K -------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(k_range, sil_scores, marker="o", color="steelblue")
    ax3.axvline(k, color="crimson", linestyle="--", label=f"K ottimale={k}")
    ax3.set_title("Silhouette score vs K")
    ax3.set_xlabel("K")
    ax3.set_ylabel("Silhouette")
    ax3.legend()

    # ---- 4. Precision-Recall per ogni specie (one-vs-rest) ------------------
    ax4 = fig.add_subplot(gs[1, :2])

    # Binarizza le label per one-vs-rest
    classes      = list(range(n_species))
    Y_true_bin   = label_binarize(win_labels,  classes=classes)
    Y_pred_bin   = label_binarize(pred_labels, classes=classes)

    if n_species == 2:
        Y_true_bin = np.hstack([1 - Y_true_bin, Y_true_bin])
        Y_pred_bin = np.hstack([1 - Y_pred_bin, Y_pred_bin])

    # Usiamo pred_labels come score continuo proxy (0/1),
    # ma per una curva piu' informativa usiamo la distanza dal centroide
    # come score di confidenza per ogni specie
    for s in range(n_species):
        # score = 1 se il cluster assegnato corrisponde alla specie s, else 0
        # (binario, la curva degenera in un punto; e' la PR "hard")
        precision, recall, _ = precision_recall_curve(
            Y_true_bin[:, s], Y_pred_bin[:, s]
        )
        ap = average_precision_score(Y_true_bin[:, s], Y_pred_bin[:, s])
        ax4.plot(recall, precision, marker=".", markersize=4,
                 label=f"{species_names[s]} (AP={ap:.2f})", color=colors[s])

    ax4.set_title("Precision-Recall per specie (one-vs-rest)")
    ax4.set_xlabel("Recall")
    ax4.set_ylabel("Precision")
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1.05])
    ax4.legend(fontsize=8)

    # ---- 5. Matrice cluster-specie (heatmap) --------------------------------
    ax5 = fig.add_subplot(gs[1, 2])
    matrix = np.zeros((k, n_species), dtype=int)
    for c in range(k):
        mask = cluster_ids == c
        for s in range(n_species):
            matrix[c, s] = int((win_labels[mask] == s).sum())

    im = ax5.imshow(matrix, aspect="auto", cmap="Blues")
    ax5.set_xticks(range(n_species))
    ax5.set_xticklabels(species_names, rotation=45, ha="right", fontsize=8)
    ax5.set_yticks(range(k))
    ax5.set_yticklabels([f"C{c}" for c in range(k)], fontsize=8)
    ax5.set_title("Finestre per cluster x specie")
    ax5.set_xlabel("Specie")
    ax5.set_ylabel("Cluster")
    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

    # Annota i valori nella heatmap
    for c in range(k):
        for s in range(n_species):
            v = matrix[c, s]
            if v > 0:
                ax5.text(s, c, str(v), ha="center", va="center",
                         fontsize=7, color="black" if v < matrix.max() * 0.6 else "white")

    fig.suptitle("Valutazione clustering spazio latente", fontsize=14, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot salvato: {save_path}")
    plt.show()


# ---- main -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  required=True)
    parser.add_argument("--model",     default="bird_model")
    parser.add_argument("--pipeline",  default=None,
                        help="Percorso pipeline .pkl (default: <model>_pipeline.pkl)")
    parser.add_argument("--k_min",     type=int, default=2)
    parser.add_argument("--k_max",     type=int, default=12)
    parser.add_argument("--plot",      action="store_true")
    parser.add_argument("--save",      default="evaluation.png")
    args = parser.parse_args()

    pipeline_path = args.pipeline or f"{args.model}_pipeline.pkl"

    # 1) Carica modelli e pipeline
    print(f"Caricamento modello: {args.model}")
    autoencoder, encoder = load_models(args.model)

    print(f"Caricamento pipeline: {pipeline_path}")
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    # 2) Raccolta file
    print(f"\nRaccolta file da: {args.data_dir}")
    paths, file_labels, species_names = collect_files(args.data_dir)
    n_species = len(species_names)
    print(f"Totale: {len(paths)} file, {n_species} specie")

    if n_species < 2:
        raise ValueError("Servono almeno 2 specie per la curva Precision-Recall.")

    # 3) Estrazione latente
    Z, win_labels = extract_latent(encoder, pipeline, paths, file_labels)

    # 4) K ottimale
    best_k, sil_scores, k_range = find_optimal_k(Z, k_min=args.k_min, k_max=args.k_max)

    # 5) Clustering e associazione specie
    print(f"\nClustering con K={best_k}...")
    cluster_ids, cluster_to_sp, pred_labels = cluster_and_assign(
        Z, win_labels, best_k, species_names
    )

    # 6) Report testuale
    print("\nClassification report (cluster -> specie):")
    print(classification_report(
        win_labels, pred_labels,
        target_names=species_names,
        zero_division=0,
    ))

    # 7) Plot
    if args.plot:
        Z2d = reduce_2d(Z)
        plot_evaluation(
            Z2d, cluster_ids, win_labels, pred_labels,
            species_names, best_k, k_range, sil_scores,
            save_path=args.save,
        )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
