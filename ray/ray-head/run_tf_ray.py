import wandb
import tensorflow as tf
import numpy as np
import os

wandb.login(key=os.environ.get("WANDB_API_KEY"))

api = wandb.Api()

artifact = api.artifact("taras-dzyk-personal/ray-tf-demo/mnist-model-128:latest", type="model")
artifact_dir = artifact.download()

model_path = os.path.join(artifact_dir, "model_128.h5")
model = tf.keras.models.load_model(model_path)

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.reshape(-1, 28 * 28) / 255.0

predictions = model.predict(x_test[:10])
predicted_classes = np.argmax(predictions, axis=1)

print("Predicted classes:", predicted_classes)
print("True classes     :", y_test[:10])
