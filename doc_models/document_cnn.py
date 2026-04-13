"""
Keras Sequential CNN for 16-class document images (RVL-CDIP-style).

Derived from the project notebook with fixes: no ReLU then LeakyReLU on the same
conv output; Dense layers ordered 4096 → 2048 → 16 (not 16 → 4096); softmax
output size matches num_classes; regularizer comment matches L2.
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
) -> models.Sequential:
    """Return an uncompiled Sequential model."""
    l2_reg = regularizers.l2(l2)
    return models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(momentum=bn_momentum),
            layers.Conv2D(16, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(8, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(rate=dropout_after_conv),
            layers.Flatten(),
            layers.Dense(
                4096,
                activation="relu",
                kernel_regularizer=l2_reg,
            ),
            layers.Dropout(rate=dropout_after_dense),
            layers.Dense(2048, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="document_cnn",
    )


if __name__ == "__main__":
    m = build_document_cnn()
    m.summary()
