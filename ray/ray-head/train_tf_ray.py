import ray
import tensorflow as tf
import numpy as np
import wandb
import time
import os
import logging
import sys

ray.init(address="auto")

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28) / 255.0
train_data_ref = ray.put((x_train, y_train))

@ray.remote
def train_model(data, hidden_units=64, epochs=3):

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info(f"[START] Training with hidden_units={hidden_units}")

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb_entity = os.environ.get("WANDB_ENTITY")


    wandb.login(key=wandb_api_key)

    wandb.init(
        project="ray-tf-demo",
        entity=wandb_entity,
        group="mnist-hidden-units",
        config={
            "hidden_units": hidden_units,
            "epochs": epochs
        },
        reinit=True
    )

    config = wandb.config
    x_train, y_train = data

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(config.hidden_units, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    start = time.time()
    history = model.fit(x_train, y_train, epochs=config.epochs, verbose=0)
    end = time.time()

    wandb.log({
        "final_accuracy": float(history.history['accuracy'][-1]),
        "duration": end - start
    })


    model_path = f"model_{config.hidden_units}.h5"
    model.save(model_path)

    artifact = wandb.Artifact(f"mnist-model-{config.hidden_units}", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    wandb.finish()

    return {
        "hidden_units": config.hidden_units,
        "accuracy": float(history.history['accuracy'][-1]),
        "duration": round(end - start, 2)
    }


futures = [
    train_model.remote(train_data_ref, hidden_units=64),
    train_model.remote(train_data_ref, hidden_units=128),
    train_model.remote(train_data_ref, hidden_units=256),
]

results = ray.get(futures)

print("Finished training:")
for r in results:
    print(r)
