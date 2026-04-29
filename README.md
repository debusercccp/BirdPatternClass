# Bird Pattern Classifier – Transformer

Identifica pattern ripetuti nei cinguettii tramite Transformer autoencoder.
Preprocessing a pipeline sklearn + inferenza in tempo reale da microfono.

---

## Architettura

```
Audio grezzo
  |
  v  AudioCleaner      noisereduce (profilo rumore automatico) + normalizzazione peak
  v  MFCCExtractor     MFCC (13) + delta + delta-delta -> (T, 39) per frame
  v  FeatureNormalizer z-score per feature (StandardScaler fittato su train)
  v  SequenceBuilder   sliding window -> (N, 64, 39)
  |
  v  Linear(d_model=128) + PositionalEncoding sinusoidale
  v  3x TransformerEncoderBlock  (MultiHeadAttention + FFN + LayerNorm)
  v  GlobalAveragePooling1D  ->  latent (128,)
  |
  v  RepeatVector(T) + PositionalEncoding
  v  3x TransformerDecoderBlock  (self-attn + cross-attn + FFN)
  v  Linear(n_features)  ->  ricostruzione (T, 39)
  |
  v  Reconstruction Error (MSE)
       basso  -> pattern ripetuto (canto familiare)
       alto   -> pattern nuovo o anomalia
```

### Perche' Transformer invece di LSTM

La self-attention confronta ogni frame con tutti gli altri nella finestra
in un singolo passo, senza degradazione dell'hidden state su distanze lunghe.
Pattern ripetuti non adiacenti vengono individuati direttamente.

| Criterio                    | Transformer       | LSTM              |
|-----------------------------|-------------------|-------------------|
| Pattern a lunga distanza    | superiore         | limitato          |
| Accuratezza (dataset grandi)| generalmente migliore | competitivo su seq. corte |
| Training                    | parallelizzabile  | sequenziale       |
| Inferenza su edge (RPi)     | piu' pesante      | piu' leggero      |
| Sequenze brevi (<32 frame)  | overhead eccessivo| preferibile       |

---

## Installazione

### Dipendenze di sistema

`sounddevice` richiede PortAudio installata a livello di sistema.
Senza di essa l'import fallisce con `OSError: PortAudio library not found`.

Debian/Ubuntu:

```bash
sudo apt install libportaudio2 portaudio19-dev
```

Arch Linux:

```bash
sudo pacman -S portaudio
```

### Dipendenze Python

```bash
pip install -r requirements.txt
```

Se PortAudio era assente al momento della prima installazione, reinstallare sounddevice dopo:

```bash
pip install --force-reinstall sounddevice
```

---

## Struttura dataset

```
dataset/
  specie_A/   *.wav / *.mp3 / *.ogg / *.flac
  specie_B/   ...
```

Cartella piatta (senza sottocartelle) = modalita' autoencoder pura, nessuna label richiesta.

---

## Training

```bash
python train.py \
  --data_dir  ./dataset \
  --epochs    60        \
  --d_model   128       \
  --num_heads 4         \
  --num_layers 3        \
  --save      bird_model \
  --plot
```

Output generato:

```
bird_model_autoencoder.keras
bird_model_encoder.keras
bird_model_pipeline.pkl
bird_model_analysis.png     (solo con --plot)
```

Al termine, train.py stampa la soglia ottimale da passare a realtime.py.

---

## Predizione realtime

```bash
# Lista dispositivi audio disponibili
python realtime.py --list-devices

# Avvio indefinito (Ctrl+C per fermare)
python realtime.py --model bird_model --threshold 0.05 --device 1 --sr 44100

# Ascolto per una durata fissa (es. 60 secondi)
python realtime.py --model bird_model --threshold 0.05 --device 1 --sr 44100 --duration 60
```

Output in console ogni ~2 s:

```
   Tempo      Errore  Stato
------------------------------------------
15:42:01     0.03120  pattern ripetuto
15:42:03     0.08754  pattern nuovo/anomalo
```

Se il device non supporta 22050 Hz (errore `Invalid sample rate`), usa `--sr 44100`
o `--sr 48000`: il preprocessing ricampiona internamente al sample rate di training.

---

## Predizione su file

```python
import pickle
from realtime import predict_file
from model import load_models

autoencoder, encoder = load_models("bird_model")

with open("bird_model_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

result = predict_file("registrazione.wav", autoencoder, pipeline, threshold_percentile=25.0)
print(result["pattern_ratio"])   # frazione di finestre con pattern ripetuto
print(result["errors"])          # MSE per ogni finestra
```

---

## Parametri principali

| Parametro            | Default | Descrizione                                      |
|----------------------|---------|--------------------------------------------------|
| `SR`                 | 22050   | Sample rate target                               |
| `N_MFCC`             | 13      | Coefficienti MFCC                                |
| `WINDOW_SIZE`        | 64      | Frame per sequenza (~1.5 s)                      |
| `WINDOW_STEP`        | 16      | Overlap tra finestre                             |
| `D_MODEL`            | 128     | Dimensione embedding Transformer                 |
| `NUM_HEADS`          | 4       | Teste di attenzione (deve dividere D_MODEL)      |
| `FF_DIM`             | 256     | Dimensione FFN interna ai blocchi                |
| `NUM_LAYERS`         | 3       | Numero di blocchi encoder e decoder              |
| `DROPOUT`            | 0.1     | Dropout nei blocchi Transformer                  |
| `threshold_percentile` | 25   | Percentile MSE sotto cui = pattern ripetuto      |
| `--sr`               | 44100   | Sample rate del device audio (realtime)          |
| `--duration`         | None    | Durata ascolto in secondi (None = Ctrl+C)        |

---

## Note su edge deployment (Raspberry Pi)

Su hardware vincolato ridurre `d_model` (es. 64), `num_layers` (es. 2)
e `num_heads` (es. 2). In alternativa usare la versione LSTM della stessa
pipeline, che ha latenza di inferenza inferiore a parita' di window_size.
