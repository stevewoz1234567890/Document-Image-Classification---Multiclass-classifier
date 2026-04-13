"""
Keras Sequential CNN for 16-class document images — matches notebook `model.summary()`.

Total params: 10,015,352 (trainable 10,015,288; non-trainable 64 from BatchNorm).
"""

from __future__ import annotations

from tensorflow.keras import layers, models, regularizers


def build_document_cnn(
    input_shape: tuple[int, int, int] = (1024, 768, 1),
    num_classes: int = 16,
    l2: float = 0.001,
    dropout_after_conv: float = 0.10,
    dropout_after_dense: float = 0.15,
    bn_momentum: float = 0.8,
    leaky_alpha_1: float = 0.2,
    leaky_alpha_2: float = 0.1,
) -> models.Sequential:
    """Return an uncompiled Sequential model (same topology as the course notebook)."""
    l2_reg = regularizers.l2(l2)
    return models.Sequential(
        [
            layers.Conv2D(32, (3, 3), input_shape=input_shape),
            layers.LeakyReLU(alpha=leaky_alpha_1),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(momentum=bn_momentum),
            layers.Conv2D(16, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(8, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(rate=dropout_after_conv),
            layers.Flatten(),
            layers.LeakyReLU(alpha=leaky_alpha_2),
            layers.Dense(
                16,
                activation="relu",
                kernel_regularizer=l2_reg,
            ),
            layers.Dropout(rate=dropout_after_dense),
            layers.Dense(4096, activation="relu"),
            layers.Dense(2048, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="document_cnn",
    )


if __name__ == "__main__":
    m = build_document_cnn()
    m.summary()
