"""
preprocessing.py – Pipeline di pulizia audio e feature extraction.

Passi:
  1. Carica audio (librosa) e riduce il rumore (noisereduce)
  2. Normalizzazione ampiezza (peak)
  3. Estrazione MFCC + delta + delta-delta -> (T, 39) per frame
  4. Z-score per feature (StandardScaler fittato su training)
  5. Sliding-window sequence builder -> (N, window_size, 39) per il Transformer
"""

import numpy as np
import librosa
import noisereduce as nr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---- iperparametri ----------------------------------------------------------
SR          = 22050
N_MFCC      = 13
HOP_LENGTH  = 512
N_FFT       = 1024
WINDOW_SIZE = 64    # frame per sequenza (~1.5 s a SR=22050, hop=512)
WINDOW_STEP = 16    # overlap tra finestre


# ---- Step 1: pulizia audio --------------------------------------------------
class AudioCleaner(BaseEstimator, TransformerMixin):
    """
    Input : lista di percorsi (str) o array numpy float32.
    Output: lista di array numpy puliti e normalizzati, shape (n_samples,).
    """

    def __init__(self, sr: int = SR, prop_decrease: float = 0.9):
        self.sr = sr
        self.prop_decrease = prop_decrease

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        results = []
        for item in X:
            if isinstance(item, str):
                audio, _ = librosa.load(item, sr=self.sr, mono=True)
            else:
                audio = np.asarray(item, dtype=np.float32)

            audio = nr.reduce_noise(
                y=audio,
                sr=self.sr,
                prop_decrease=self.prop_decrease,
                stationary=False,
            )

            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak

            results.append(audio)
        return results


# ---- Step 2: estrazione MFCC ------------------------------------------------
class MFCCExtractor(BaseEstimator, TransformerMixin):
    """
    Input : lista di array audio float32.
    Output: lista di (T_frames, n_mfcc * 3).
    """

    def __init__(
        self,
        sr: int = SR,
        n_mfcc: int = N_MFCC,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
    ):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        results = []
        for audio in X:
            mfcc   = librosa.feature.mfcc(
                y=audio, sr=self.sr, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length,
            )
            delta  = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            features = np.vstack([mfcc, delta, delta2]).T  # (T, 39)
            results.append(features)
        return results


# ---- Step 3: normalizzazione per feature ------------------------------------
class FeatureNormalizer(BaseEstimator, TransformerMixin):
    """
    Fit su tutti i frame del training set, poi z-score per feature.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        all_frames = np.vstack(X)
        self.scaler.fit(all_frames)
        return self

    def transform(self, X, y=None):
        return [self.scaler.transform(seq) for seq in X]


# ---- Step 4: sliding window -------------------------------------------------
class SequenceBuilder(BaseEstimator, TransformerMixin):
    """
    Converte lista di (T, F) in array (N, window_size, F).
    Gli indici originali (file_idx, frame_start) sono in self.origins_
    dopo transform.
    """

    def __init__(self, window_size: int = WINDOW_SIZE, step: int = WINDOW_STEP):
        self.window_size = window_size
        self.step = step
        self.origins_ = []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        windows = []
        self.origins_ = []

        for file_idx, seq in enumerate(X):
            T = seq.shape[0]
            for start in range(0, T - self.window_size + 1, self.step):
                windows.append(seq[start : start + self.window_size])
                self.origins_.append((file_idx, start))

        if not windows:
            raise ValueError(
                "Nessuna sequenza estratta: audio troppo corto rispetto a window_size."
            )

        return np.stack(windows)  # (N, window_size, F)


# ---- Pipeline completa -------------------------------------------------------
def build_preprocessing_pipeline(
    sr: int = SR,
    n_mfcc: int = N_MFCC,
    window_size: int = WINDOW_SIZE,
    window_step: int = WINDOW_STEP,
) -> Pipeline:
    """
    Ritorna la pipeline sklearn pronta per fit_transform su lista di file.

    Esempio:
        pipe = build_preprocessing_pipeline()
        X = pipe.fit_transform(file_paths)  # (N, window_size, 39)
    """
    return Pipeline([
        ("cleaner",    AudioCleaner(sr=sr)),
        ("mfcc",       MFCCExtractor(sr=sr, n_mfcc=n_mfcc)),
        ("normalizer", FeatureNormalizer()),
        ("sequences",  SequenceBuilder(window_size=window_size, step=window_step)),
    ])
