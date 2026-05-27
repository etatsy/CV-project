from __future__ import annotations

import json
import os
import time

import sys

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

sys.path.insert(0, os.path.dirname(__file__))
from local_cifar10 import load_cifar10_from_tar  # noqa: E402


def train() -> dict:
    os.makedirs("artifacts", exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.abspath("artifacts/mplconfig"))

    data = load_cifar10_from_tar(".keras/datasets/cifar-10-python.tar.gz")
    x_train, y_train = data.x_train[:10_000], data.y_train[:10_000]
    x_test, y_test = data.x_test, data.y_test

    x_train_pp = preprocess_input(x_train.astype("float32"))
    x_test_pp = preprocess_input(x_test.astype("float32"))

    base_model = ResNet50(
        weights=".keras/models/resnet50_notop.h5",
        include_top=False,
        input_shape=(32, 32, 3),
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(32, 32, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    ]

    def train_stage(lr: float, epochs: int, tag: str) -> dict:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        t0 = time.time()
        history = model.fit(
            x_train_pp,
            y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=64,
            callbacks=callbacks,
            verbose=2,
        )
        seconds = time.time() - t0

        hist = history.history
        with open(f"artifacts/history_{tag}.json", "w") as f:
            json.dump(hist, f, indent=2)

        return {
            "tag": tag,
            "epochs_ran": len(hist.get("loss", [])),
            "train_acc_last": float(hist.get("accuracy", [float("nan")])[-1]),
            "val_acc_best": float(max(hist.get("val_accuracy", [float("nan")]))),
            "val_loss_best": float(min(hist.get("val_loss", [float("nan")]))),
            "seconds": float(seconds),
        }

    stage1 = train_stage(lr=1e-3, epochs=10, tag="head")

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    stage2 = train_stage(lr=1e-5, epochs=2, tag="finetune_last20")

    test_loss, test_acc = model.evaluate(x_test_pp, y_test, verbose=0)

    summary = {
        "tensorflow": tf.__version__,
        "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        "stage1": stage1,
        "stage2": stage2,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    }

    with open("artifacts/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
