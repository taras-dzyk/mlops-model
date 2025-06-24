import ray
import tensorflow as tf
import numpy as np
import wandb
import time
import os
import logging
import sys

ray.init(address="auto")


print(f'Preparing a model')

import tensorflow as tf
import pathlib

data_dir = pathlib.Path('dataset') 


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    label_mode='int',
    image_size=(224, 224),     
    batch_size=None,         
    shuffle=True
)

# Перетворимо багатокласові мітки у бінарні: 1 якщо "crocodiles", інакше 0
class_names = dataset.class_names  
print("Мапінг класів:", class_names)

crocodile_index = class_names.index("crocodiles")

def to_binary_label(image, label):
    binary_label = tf.cast(tf.equal(label, crocodile_index), tf.int32)
    return image, binary_label

dataset_binary = dataset.map(to_binary_label)


# --------------------
# Розбиття на train/val
# --------------------
dataset_binary = dataset_binary.shuffle(1000).cache()


dataset_size = dataset_binary.cardinality().numpy()
train_size = int(0.8 * dataset_size)

train_ds = dataset_binary.take(train_size).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = dataset_binary.skip(train_size).batch(32).prefetch(tf.data.AUTOTUNE)

def dataset_to_numpy(ds):
    x_list = []
    y_list = []
    for x_batch, y_batch in ds:
        x_list.append(x_batch.numpy())
        y_list.append(y_batch.numpy())
    return (
        np.concatenate(x_list, axis=0),
        np.concatenate(y_list, axis=0)
    )

train_x, train_y = dataset_to_numpy(train_ds)
val_x, val_y = dataset_to_numpy(val_ds)

train_val_dataset_ref = ray.put(((train_x, train_y), (val_x, val_y)))

@ray.remote
def train_model(data, hidden_units=64, epochs=1):

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
        project="ray-img-demo",
        entity=wandb_entity,
        #group="mnist-hidden-units",
        config={
            "hidden_units": hidden_units,
            "epochs": epochs
        },
        reinit=True
    )

    config = wandb.config
    (train_x, train_y), (val_x, val_y) = data

    # --------------------
    # Побудова бінарної моделі
    # --------------------
    import tensorflow as tf
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='models/weights.h5'
    )
    base_model.trainable = False 

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model_binary = tf.keras.Model(inputs, outputs)

    model_binary.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    model_binary.summary()



    start = time.time()



    # --------------------
    # Навчання
    # --------------------
    neg = np.sum(train_y == 0)
    pos = np.sum(train_y == 1)
    total = neg + pos

    weight_for_0 = total / (2 * neg)
    weight_for_1 = total / (2 * pos)

    class_weights = {0: weight_for_0, 1: weight_for_1}
    print("Class weights:", class_weights)

    history = model_binary.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=epochs,
        class_weight=class_weights
    )


    # --------------------
    # Оцінка
    # --------------------
    from sklearn.metrics import classification_report

    def evaluate_binary_model(model, x_val, y_val, batch_size=32):
        y_pred = model.predict(x_val, batch_size=batch_size).squeeze()
        y_pred_binary = (y_pred > 0.5).astype(int)

        print(classification_report(y_val, y_pred_binary, digits=3))



    evaluate_binary_model(model_binary, val_x, val_y)
    
    
    end = time.time()

    wandb.log({
        "final_accuracy": float(history.history['accuracy'][-1]),
        "duration": end - start
    })


    model_path = f"model_{config.hidden_units}.keras"
    model_binary.save(model_path)

    artifact = wandb.Artifact(f"crockodile-finder-model-{config.hidden_units}", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)



    wandb.finish()

    return {
        "hidden_units": config.hidden_units,
        "accuracy": float(history.history['accuracy'][-1]),
        "duration": round(end - start, 2)
    }


futures = [
    train_model.remote(train_val_dataset_ref, hidden_units=128),
]

results = ray.get(futures)

print("Finished training:")
for r in results:
    print(r)
