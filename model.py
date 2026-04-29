"""
model.py – Transformer autoencoder per rilevamento pattern audio.

Architettura:
  Encoder
    Linear(d_model) + PositionalEncoding
    N x TransformerEncoderLayer (MultiHeadAttention + FFN + LayerNorm)
    GlobalAveragePooling1D  ->  latent (d_model,)

  Decoder
    RepeatVector(T)
    N x TransformerDecoderLayer (masked self-attn + cross-attn + FFN)
    Linear(n_features)  ->  ricostruzione (T, F)

Perche' Transformer vs LSTM:
  - La self-attention confronta ogni frame con tutti gli altri in O(1) hop:
    pattern ripetuti vengono individuati anche a distanza arbitraria.
  - Training parallelizzabile (nessuna dipendenza sequenziale).
  - Su sequenze lunghe (>64 frame) supera nettamente i modelli ricorrenti.
  Svantaggio: piu' pesante in inferenza; su edge (Raspberry Pi) valutare
  WINDOW_SIZE ridotto o distillazione.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Optional, Tuple


# ---- iperparametri default ---------------------------------------------------
D_MODEL      = 128   # dimensione embedding interno
NUM_HEADS    = 4     # teste di attenzione (deve dividere D_MODEL)
FF_DIM       = 256   # dimensione FFN interna al blocco Transformer
NUM_LAYERS   = 3     # blocchi Transformer (encoder e decoder)
DROPOUT      = 0.1
LEARNING_RATE = 1e-3


# ---- Positional Encoding sinusoidale ----------------------------------------
class PositionalEncoding(layers.Layer):
    """
    Aggiunge codifica posizionale sinusoidale alle embedding.
    Identica alla formulazione originale (Vaswani et al. 2017).
    """

    def __init__(self, max_len: int = 512, d_model: int = D_MODEL, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

        positions = np.arange(max_len)[:, np.newaxis]           # (max_len, 1)
        dims      = np.arange(d_model)[np.newaxis, :]           # (1, d_model)
        angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)

        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        self._pe = tf.constant(angles[np.newaxis, :, :], dtype=tf.float32)  # (1, max_len, d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self._pe[:, :seq_len, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model})
        return cfg


# ---- singolo blocco Transformer Encoder -------------------------------------
class TransformerEncoderBlock(layers.Layer):
    """
    MultiHeadAttention -> Add&Norm -> FFN -> Add&Norm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = DROPOUT,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attn   = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn    = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(d_model),
        ])
        self.norm1  = layers.LayerNormalization(epsilon=1e-6)
        self.norm2  = layers.LayerNormalization(epsilon=1e-6)
        self.drop1  = layers.Dropout(dropout)
        self.drop2  = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.attn(x, x, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ffn_out  = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out, training=training))
        return x


# ---- singolo blocco Transformer Decoder -------------------------------------
class TransformerDecoderBlock(layers.Layer):
    """
    Masked self-attn -> cross-attn (su encoder output) -> FFN, con Add&Norm.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = DROPOUT,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.self_attn  = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.cross_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)
        self.drop3 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        # inputs e' una lista [x, encoder_out] perche' Keras non fa
        # unpacking automatico quando il layer viene chiamato con piu' tensori
        x, encoder_out = inputs

        sa = self.self_attn(x, x, training=training)
        x  = self.norm1(x + self.drop1(sa, training=training))

        ca = self.cross_attn(x, encoder_out, training=training)
        x  = self.norm2(x + self.drop2(ca, training=training))

        ff = self.ffn(x)
        x  = self.norm3(x + self.drop3(ff, training=training))
        return x


# ---- Autoencoder Transformer ------------------------------------------------
def build_autoencoder(
    window_size: int,
    n_features: int,
    d_model: int = D_MODEL,
    num_heads: int = NUM_HEADS,
    ff_dim: int = FF_DIM,
    num_layers: int = NUM_LAYERS,
    dropout: float = DROPOUT,
    n_classes: Optional[int] = None,
) -> Tuple[Model, Model]:
    """
    Costruisce Transformer autoencoder.

    Args:
        window_size : frame per sequenza (T).
        n_features  : feature per frame (es. 39).
        n_classes   : se fornito, aggiunge testa classificazione sul latent.

    Returns:
        autoencoder, encoder
        oppure (autoencoder, encoder, classifier) se n_classes e' fornito.
    """
    inp = Input(shape=(window_size, n_features), name="input_seq")

    # ---- Encoder -----------------------------------------------------------
    # Proiezione lineare verso d_model
    x = layers.Dense(d_model, name="input_proj")(inp)
    x = PositionalEncoding(max_len=window_size, d_model=d_model, name="pos_enc_enc")(x)
    x = layers.Dropout(dropout, name="drop_enc_in")(x)

    for i in range(num_layers):
        x = TransformerEncoderBlock(
            d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout,
            name=f"enc_block_{i}"
        )(x)

    # Latent: media su tutta la sequenza (sequence-level representation)
    latent = layers.GlobalAveragePooling1D(name="latent")(x)   # (batch, d_model)
    enc_out = x   # (batch, T, d_model) – usato come encoder_out nel decoder

    # ---- Decoder -----------------------------------------------------------
    # Espandi il latent a (batch, T, d_model) come query iniziale del decoder
    dec_in = layers.RepeatVector(window_size, name="repeat_latent")(latent)
    dec_in = PositionalEncoding(max_len=window_size, d_model=d_model, name="pos_enc_dec")(dec_in)

    # I blocchi decoder hanno bisogno sia di dec_in che di enc_out:
    # usiamo un wrapper funzionale esplicito
    dec_x = dec_in
    for i in range(num_layers):
        dec_x = TransformerDecoderBlock(
            d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout,
            name=f"dec_block_{i}"
        )([dec_x, enc_out])    # Keras accetta lista di input per layer custom

    reconstruction = layers.TimeDistributed(
        layers.Dense(n_features), name="reconstruction"
    )(dec_x)  # (batch, T, n_features)

    # ---- Modelli ------------------------------------------------------------
    encoder     = Model(inp, latent, name="encoder")
    autoencoder = Model(inp, reconstruction, name="autoencoder")

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )

    if n_classes is not None:
        cls_out    = layers.Dense(n_classes, activation="softmax", name="class_out")(latent)
        classifier = Model(inp, cls_out, name="classifier")
        classifier.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return autoencoder, encoder, classifier

    return autoencoder, encoder


# ---- Training ---------------------------------------------------------------
def train_autoencoder(
    model: Model,
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    epochs: int = 50,
    batch_size: int = 32,
) -> tf.keras.callbacks.History:
    callbacks = [
        EarlyStopping(patience=7, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, monitor="val_loss"),
    ]
    val_data = (X_val, X_val) if X_val is not None else None
    return model.fit(
        X_train, X_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )


# ---- Rilevamento pattern ----------------------------------------------------
def compute_reconstruction_error(model: Model, X: np.ndarray) -> np.ndarray:
    """
    MSE per ogni finestra.
      - errore basso  -> pattern familiare/ripetuto
      - errore alto   -> pattern nuovo o anomalia
    """
    X_pred = model.predict(X, verbose=0)
    return np.mean(np.square(X - X_pred), axis=(1, 2))


def detect_repeated_patterns(
    model: Model,
    X: np.ndarray,
    threshold_percentile: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Identifica finestre con pattern ripetuto (basso reconstruction error).

    Returns:
        errors     : (N,) MSE per finestra
        is_pattern : (N,) bool -- True = pattern ripetuto
        threshold  : soglia usata
    """
    errors    = compute_reconstruction_error(model, X)
    threshold = np.percentile(errors, threshold_percentile)
    return errors, errors <= threshold, threshold


# ---- I/O modelli ------------------------------------------------------------
def save_models(autoencoder: Model, encoder: Model, prefix: str = "bird_model"):
    autoencoder.save(f"{prefix}_autoencoder.keras")
    encoder.save(f"{prefix}_encoder.keras")
    print(f"Salvati: {prefix}_autoencoder.keras, {prefix}_encoder.keras")


def load_models(prefix: str = "bird_model") -> Tuple[Model, Model]:
    custom = {
        "PositionalEncoding":       PositionalEncoding,
        "TransformerEncoderBlock":  TransformerEncoderBlock,
        "TransformerDecoderBlock":  TransformerDecoderBlock,
    }
    autoencoder = tf.keras.models.load_model(f"{prefix}_autoencoder.keras", custom_objects=custom)
    encoder     = tf.keras.models.load_model(f"{prefix}_encoder.keras",     custom_objects=custom)
    return autoencoder, encoder
