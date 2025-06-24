import ray
import time

ray.init(address="auto")

@ray.remote
def square(x):
    time.sleep(1)
    return x * x

start = time.time()
results = ray.get([square.remote(i) for i in range(8)])
end = time.time()

print("Results:", results)
print(f"Took {end - start:.2f} seconds")
