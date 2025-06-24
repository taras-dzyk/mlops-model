import os
import subprocess

dst = "dataset"
os.makedirs(dst, exist_ok=True)

print("`dvc pull -r localstorage`")
result = subprocess.run(["dvc", "pull", "dataset", "-r", "localstorage"], check=True)

print("датасет отримано")
