"""
realtime.py – Rilevamento pattern in tempo reale da microfono.

Cattura audio con sounddevice, lo processa con la stessa pipeline
del training, e calcola il reconstruction error del Transformer
per ogni finestra. Finestre con errore basso corrispondono a
pattern gia' visti (ripetuti).

Avvio:
    python realtime.py --model bird_model --threshold 0.05
    python realtime.py --list-devices
"""

import argparse
import threading
import time
import queue
from collections import deque
from typing import Optional

import numpy as np
import noisereduce as nr
import librosa
import tensorflow as tf
import sounddevice as sd

from preprocessing import SR, N_MFCC, N_FFT, HOP_LENGTH, WINDOW_SIZE
from model import load_models, compute_reconstruction_error


# ---- Buffer thread-safe ------------------------------------------------------
class RingBuffer:
    def __init__(self, maxlen: int):
        self._buf  = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def extend(self, data: np.ndarray):
        with self._lock:
            self._buf.extend(data.flatten())

    def get_copy(self) -> np.ndarray:
        with self._lock:
            return np.array(self._buf, dtype=np.float32)

    def __len__(self):
        with self._lock:
            return len(self._buf)


# ---- Preprocessing su chunk singolo -----------------------------------------
def preprocess_chunk(
    audio: np.ndarray,
    sr: int = SR,
    n_mfcc: int = N_MFCC,
    scaler=None,
) -> Optional[np.ndarray]:
    """
    Pulisce e trasforma un chunk grezzo in (1, WINDOW_SIZE, n_mfcc*3).
    Ritorna None se il chunk e' troppo corto o silenzioso.
    """
    audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8, stationary=True)

    # ricampiona al SR di training se il device usa un rate diverso
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        sr = SR

    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return None
    audio = audio / peak

    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2]).T  # (T, 39)

    if features.shape[0] < WINDOW_SIZE:
        return None

    window = features[-WINDOW_SIZE:]

    if scaler is not None:
        window = scaler.transform(window)

    return window[np.newaxis, ...]  # (1, WINDOW_SIZE, 39)


# ---- Rilevatore realtime -----------------------------------------------------
class BirdPatternDetector:
    """
    Cattura audio in streaming e predice pattern ripetuti in tempo reale.

    Uso:
        detector = BirdPatternDetector(autoencoder, encoder, scaler, threshold=0.05)
        detector.start()   # blocca fino a Ctrl+C
    """

    def __init__(
        self,
        autoencoder: tf.keras.Model,
        encoder: tf.keras.Model,
        scaler=None,
        sr: int = SR,
        chunk_duration: float = 2.0,
        threshold: float = 0.05,
        device: Optional[int] = None,
        duration: Optional[float] = None,
    ):
        self.autoencoder   = autoencoder
        self.encoder       = encoder
        self.scaler        = scaler
        self.sr            = sr
        self.chunk_samples = int(sr * chunk_duration)
        self.threshold     = threshold
        self.device        = device

        self.duration  = duration
        self._buffer  = RingBuffer(maxlen=self.chunk_samples * 4)
        self._q       = queue.Queue()
        self._running = False
        self._stream  = None
        self._thread  = None

        self.error_history: list[float] = []

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"sounddevice status: {status}")
        self._buffer.extend(indata[:, 0])
        if len(self._buffer) >= self.chunk_samples:
            self._q.put(self._buffer.get_copy()[-self.chunk_samples:])

    def _inference_loop(self):
        print(f"\nAscolto in corso... (soglia errore: {self.threshold:.4f})")
        print(f"{'Tempo':>8}  {'Errore':>10}  Stato")
        print("-" * 42)

        while self._running:
            try:
                chunk = self._q.get(timeout=1.0)
            except queue.Empty:
                continue

            seq = preprocess_chunk(chunk, sr=self.sr, scaler=self.scaler)
            if seq is None:
                continue

            error      = compute_reconstruction_error(self.autoencoder, seq)[0]
            is_pattern = error <= self.threshold
            stato      = "pattern ripetuto" if is_pattern else "pattern nuovo/anomalo"
            t          = time.strftime("%H:%M:%S")
            print(f"{t:>8}  {error:>10.5f}  {stato}")
            self.error_history.append(error)

    def start(self):
        self._running = True

        # Rileva automaticamente canali e sample rate nativi del device
        dev_info  = sd.query_devices(self.device, kind="input")
        n_channels = int(dev_info["max_input_channels"])
        native_sr  = int(dev_info["default_samplerate"])
        if n_channels == 0:
            raise RuntimeError(f"Il device selezionato non ha canali di input: {dev_info['name']}")
        # Usa il sample rate nativo se non e' stato forzato dall'utente
        if self.sr != native_sr:
            print(f"Nota: device nativo a {native_sr} Hz, ricampionamento a {SR} Hz attivo.")
        self.sr = native_sr

        print(f"Device  : {dev_info['name']}")
        print(f"Canali  : {n_channels}")
        print(f"SR      : {native_sr} Hz")

        self._stream  = sd.InputStream(
            samplerate=native_sr,
            channels=n_channels,
            dtype="float32",
            blocksize=int(native_sr * 0.1),
            device=self.device,
            callback=self._audio_callback,
        )
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._stream.start()
        self._thread.start()

        try:
            if self.duration is not None:
                time.sleep(self.duration)
                self.stop()
            else:
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        if self._thread:
            self._thread.join(timeout=2.0)
        print("\nRilevazione fermata.")
        if self.error_history:
            arr = self.error_history
            print(f"  Media errore: {np.mean(arr):.5f}")
            print(f"  Min: {np.min(arr):.5f}  Max: {np.max(arr):.5f}")


# ---- Predizione su file singolo ---------------------------------------------
def predict_file(
    filepath: str,
    autoencoder: tf.keras.Model,
    pipeline,
    threshold_percentile: float = 25.0,
) -> dict:
    """
    Predici pattern ripetuti su un file audio.

    Args:
        threshold_percentile: percentile di errore sotto cui si considera
                              "pattern ripetuto" (default 25).

    Returns:
        dict con n_windows, n_patterns, pattern_ratio, threshold, errors, is_pattern.
    """
    from model import detect_repeated_patterns

    X = pipeline.transform([filepath])
    errors, is_pattern, thr = detect_repeated_patterns(
        autoencoder, X, threshold_percentile=threshold_percentile
    )

    result = {
        "n_windows":     len(errors),
        "n_patterns":    int(is_pattern.sum()),
        "pattern_ratio": float(is_pattern.mean()),
        "threshold":     float(thr),
        "errors":        errors,
        "is_pattern":    is_pattern,
    }

    print(f"\n{filepath}")
    print(f"  Finestre totali : {result['n_windows']}")
    print(f"  Pattern ripetuti: {result['n_patterns']} ({result['pattern_ratio']*100:.1f}%)")
    print(f"  Soglia errore   : {result['threshold']:.5f}")
    return result


# ---- Entry point ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bird pattern detector - realtime")
    parser.add_argument("--model",        default="bird_model")
    parser.add_argument("--threshold",    type=float, default=0.05)
    parser.add_argument("--device",       type=int,   default=1,
                        help="ID device audio (default 1)")
    parser.add_argument("--sr",           type=int,   default=44100,
                        help="Sample rate del device (default 44100). Ricampionato a SR di training internamente.")
    parser.add_argument("--duration",     type=float, default=None,
                        help="Durata ascolto in secondi (default: indefinita fino a Ctrl+C)")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        raise SystemExit(0)

    print(f"Caricamento modello: {args.model}_autoencoder.keras")
    autoencoder, encoder = load_models(args.model)

    detector = BirdPatternDetector(
        autoencoder=autoencoder,
        encoder=encoder,
        scaler=None,
        sr=args.sr,
        threshold=args.threshold,
        device=args.device,
        duration=args.duration,
    )
    detector.start()
