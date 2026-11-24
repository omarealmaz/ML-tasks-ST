#!/home/omarealmaz/tf-env/bin/python3
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Load the MNIST dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Training set: {x_train.shape[0]} images ({x_train.shape[1]}x{x_train.shape[2]})")
print(f"Test set:     {x_test.shape[0]} images ({x_test.shape[1]}x{x_test.shape[2]})")

# 2. Display sample images in a 3x5 grid
num_rows, num_cols = 3, 5
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6))
fig.suptitle("MNIST Samples (tf.keras.datasets.mnist)", fontsize=14)

for i in range(num_rows * num_cols):
    ax = axes[i // num_cols, i % num_cols]
    ax.imshow(x_train[i], cmap='gray')
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis('off')

plt.tight_layout()

# 3. Save preview image and display window
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
output_path = os.path.join(PROJECT_ROOT, "assets", "mnist_samples.png")
plt.savefig(output_path)
print(f"Preview saved to {output_path}")

plt.show()
