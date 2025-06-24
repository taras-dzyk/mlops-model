import wandb
import tensorflow as tf
import numpy as np
import os
import shutil
from PIL import Image
from pathlib import Path
import ray
import tempfile
from prometheus_client import Counter, Histogram, start_http_server
import time

wandb.login(key=os.environ.get("WANDB_API_KEY"))
api = wandb.Api()

artifact = api.artifact("taras-dzyk-personal/ray-img-demo/crockodile-finder-model-128:latest", type="model")
artifact_dir = artifact.download()

print("Model downloaded to:", artifact_dir)

model_path = os.path.join(artifact_dir, "model_128.keras")
print("Full model path:", model_path)

start_http_server(8000)
print("Prometheus server started on port 8000")

PREDICTIONS_TOTAL = Counter("predictions_total", "Загальна кількість передбачень")
PREDICTIONS_POSITIVE = Counter("predictions_positive_total", "Позитивні передбачення")
PREDICTION_TIME = Histogram("prediction_duration_seconds", "Тривалість передбачення")

def collect_images(input_dir="input"):
    input_dir = Path(input_dir)
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

    def is_image_file(path):
        return path.suffix.lower() in image_extensions

    return [img for img in input_dir.glob("*") if img.is_file() and is_image_file(img)]

def load_and_preprocess(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img)
    return img_array

@ray.remote
class Predictor:
    def __init__(self, model_bytes):
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp.write(model_bytes)
            self.temp_model_path = tmp.name
        self.model = tf.keras.models.load_model(self.temp_model_path)

    def predict(self, images_batch, filenames_batch, probability=0.5):
        start_time = time.time()
        matched = []
        imgs_tensor = tf.convert_to_tensor(images_batch)
        preds = self.model.predict(imgs_tensor, verbose=0)
        for filename, pred in zip(filenames_batch, preds):
            score = float(pred[0])
            if score > probability:
                matched.append((filename, score))
        duration = time.time() - start_time
        return matched, duration

def copy_matched_images(matched, output_dir="output", input_dir="input"):
    output_dir = Path(output_dir)
    input_dir = Path(input_dir)
    output_dir.mkdir(exist_ok=True)

    for filename, score in matched:
        src_path = input_dir / filename
        dst_path = output_dir / filename
        if src_path.exists():
            shutil.copy(src_path, dst_path)
            print(f"Знайдено крокодила! {filename} (score: {score:.3f})")

def main_inference_loop(model_path, probability=0.5, batch_size=8, input_dir="input", check_interval=10):
    ray.init(runtime_env={"working_dir": artifact_dir}, ignore_reinit_error=True)

    with open(model_path, "rb") as f:
        model_bytes = f.read()
    model_bytes_ref = ray.put(model_bytes)

    predictor = Predictor.remote(model_bytes_ref)

    print("Prometheus server started on port 8000")
    print("Починаємо моніторинг папки і обробку нових зображень...")

    while True:
        image_paths = collect_images(input_dir)
        if not image_paths:
            time.sleep(check_interval)
            continue

        print(f"Знайдено {len(image_paths)} зображень для обробки")

        batches = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = [load_and_preprocess(p) for p in batch_paths]
            filenames = [p.name for p in batch_paths]
            batches.append((images, filenames))

        futures = []
        for images_batch, filenames_batch in batches:
            futures.append(predictor.predict.remote(images_batch, filenames_batch, probability))

        matched = []
        total_duration = 0.0
        for future in futures:
            batch_matched, duration = ray.get(future)
            matched.extend(batch_matched)
            total_duration += duration

            PREDICTIONS_TOTAL.inc(batch_size)
            PREDICTIONS_POSITIVE.inc(len(matched))
        if futures:
            avg_duration = total_duration / len(futures)
            PREDICTION_TIME.observe(avg_duration)

        copy_matched_images(matched, output_dir="output", input_dir=input_dir)

        # removing files
        for p in image_paths:
            try:
                p.unlink()
            except Exception as e:
                print(f"Не вдалося видалити {p}: {e}")

        print(f"Оброблено {len(image_paths)} зображень, знайдено крокодилів: {len(matched)}")
        time.sleep(check_interval)


  
main_inference_loop(model_path=model_path, probability=0.5, batch_size=8)
