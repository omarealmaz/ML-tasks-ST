#!/home/omarealmaz/tf-env/bin/python3
import sys
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from segmentation import segment_and_preprocess

def predict_from_image(image_path, model_path="mnist_cnn.keras"):
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: '{image_path}'")
        print("💡 Tip: If your filename contains spaces, remember to wrap it in quotes:")
        print(f"   python3 predict.py \"{image_path}\"")
        return None

    # Load trained model
    print(f"Loading CNN model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Segment image into multiple digits
    print(f"Scanning and segmenting '{image_path}' for digits...")
    digits_data = segment_and_preprocess(image_path)

    if not digits_data:
        print("⚠️ No distinct digits detected in the image.")
        return None

    print(f"\n✅ Found {len(digits_data)} digit(s) in image (sorted left-to-right):\n")
    
    recognized_sequence = []
    
    for idx, item in enumerate(digits_data):
        tensor = item['tensor']
        x, y, w, h = item['bbox']
        
        # Predict
        probs = model.predict(tensor, verbose=0)[0]
        digit = int(np.argmax(probs))
        confidence = float(probs[digit]) * 100
        recognized_sequence.append(str(digit))

        print(f"--- Digit #{idx + 1} ---")
        print(f"Bounding Box: x={x}, y={y}, w={w}, h={h}")
        print(f"Prediction:   {digit}  ({confidence:.2f}% confidence)")
        
        # Display mini ASCII preview of what the network saw
        preview_arr = item['crop_array']
        ascii_chars = " .:-=+*#%@"
        print("Network Input (28x28 normalized):")
        for row in range(0, 28, 2):  # Skip alternate rows for square aspect ratio in terminal
            line = "".join(ascii_chars[int(preview_arr[row, col] * (len(ascii_chars) - 1))] for col in range(28))
            print("  " + line)
        print()

    full_number = "".join(recognized_sequence)
    print("====================================================")
    print(f"🎯 Full Recognized Number Sequence:  {full_number}")
    print("====================================================")
    return full_number

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <image_path>")
        print("Example: python3 predict.py image.png")
        sys.exit(1)
        
    # Join arguments to handle unquoted filenames with spaces like `image copy.png`
    img_arg = " ".join(sys.argv[1:])
    predict_from_image(img_arg)
