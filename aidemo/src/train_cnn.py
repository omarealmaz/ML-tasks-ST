#!/home/omarealmaz/tf-env/bin/python3
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def build_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Constructs a CNN model consisting of:
    - 3 Convolutional Layers (with ReLU activation and Max-Pooling)
    - 2 Fully Connected (Dense) Layers (1 feature representation + 1 output classification)
    """
    model = models.Sequential([
        # 1st Convolution Layer
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv_layer_1'),
        layers.MaxPooling2D(pool_size=(2, 2), name='maxpool_1'),

        # 2nd Convolution Layer
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv_layer_2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='maxpool_2'),

        # 3rd Convolution Layer
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv_layer_3'),

        # Flatten features for fully connected layers
        layers.Flatten(name='flatten'),

        # 1st Fully Connected Layer
        layers.Dense(64, activation='relu', name='fc_layer_1'),
        layers.Dropout(0.25, name='dropout'),

        # 2nd Fully Connected Layer (Output Layer)
        layers.Dense(num_classes, activation='softmax', name='fc_layer_2_output')
    ], name='mnist_3conv_2fc_cnn')

    return model

def main():
    # 1. Load MNIST dataset
    print("Loading MNIST dataset...")
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values to [0, 1] and add channel dimension (28, 28, 1)
    x_train_full = np.expand_dims(x_train_full.astype('float32') / 255.0, -1)
    x_test = np.expand_dims(x_test.astype('float32') / 255.0, -1)

    # 2. Partition training set into Training (50,000) and Validation (10,000)
    val_size = 10000
    x_val, y_val = x_train_full[:val_size], y_train_full[:val_size]
    x_train, y_train = x_train_full[val_size:], y_train_full[val_size:]

    print(f"Data partitioning:")
    print(f"  - Training samples:   {x_train.shape[0]}")
    print(f"  - Validation samples: {x_val.shape[0]}")
    print(f"  - Test samples:       {x_test.shape[0]}")

    # 3. Build & Compile CNN Model
    model = build_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 4. Train Model with Validation
    print("\nTraining CNN model for 6 epochs...")
    history = model.fit(
        x_train, y_train,
        epochs=6,
        batch_size=64,
        validation_data=(x_val, y_val),
        verbose=1
    )

    # 5. Evaluate on unseen Test Set
    print("\nEvaluating on Test dataset...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"--> Test Loss:     {test_loss:.4f}")
    print(f"--> Test Accuracy: {test_accuracy * 100:.2f}%")

    # 6. Save Model
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "mnist_cnn.keras")
    model.save(model_save_path)
    print(f"\nModel saved successfully to {model_save_path}")

    # 7. Plot and Save Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
    ax1.set_title('Training & Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Val Loss', marker='s')
    ax2.set_title('Training & Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    assets_dir = os.path.join(PROJECT_ROOT, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    curves_file = os.path.join(assets_dir, "training_curves.png")
    plt.savefig(curves_file)
    plt.close()
    print(f"Training curves saved to {curves_file}")

    # 8. Generate Visual Test Predictions
    predictions = model.predict(x_test[:15])
    predicted_labels = np.argmax(predictions, axis=1)

    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    fig.suptitle("Model Predictions on Test Samples", fontsize=14)

    for i in range(15):
        ax = axes[i // 5, i % 5]
        ax.imshow(x_test[i].squeeze(), cmap='gray')
        is_correct = (predicted_labels[i] == y_test[i])
        color = 'green' if is_correct else 'red'
        ax.set_title(f"Pred: {predicted_labels[i]} (True: {y_test[i]})", color=color)
        ax.axis('off')

    plt.tight_layout()
    pred_file = os.path.join(assets_dir, "predictions_sample.png")
    plt.savefig(pred_file)
    plt.close()
    print(f"Prediction samples saved to {pred_file}")

if __name__ == '__main__':
    main()
