#!/home/omarealmaz/tf-env/bin/python3
import os
import sys
import io
import json
import base64
import http.server
import socketserver
import webbrowser
import numpy as np
import tensorflow as tf
from PIL import Image
from segmentation import segment_and_preprocess

# Default port with fallback
START_PORT = 8000
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_cnn.keras")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
    sys.exit(1)

print(f"Loading CNN model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# Feature activation extractor
activation_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        model.get_layer('conv_layer_1').output,
        model.get_layer('conv_layer_2').output,
        model.get_layer('conv_layer_3').output,
        model.get_layer('fc_layer_1').output,
        model.get_layer('fc_layer_2_output').output
    ]
)

def array_to_base64(arr):
    """Converts a 2D float array in [0, 1] to a base64 PNG data URL."""
    norm = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(norm)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

def analyze_number(digits_seq):
    """Calculates mathematical insights on the recognized sequence."""
    if not digits_seq:
        return {}
    try:
        num = int("".join(map(str, digits_seq)))
    except ValueError:
        return {}

    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    digits_list = [int(d) for d in digits_seq]
    sum_digits = sum(digits_list)
    prod_digits = 1
    for d in digits_list:
        prod_digits *= d

    return {
        "value": num,
        "is_even": (num % 2 == 0),
        "is_prime": is_prime(num),
        "sum_of_digits": sum_digits,
        "product_of_digits": prod_digits,
        "binary": bin(num)[2:],
        "hex": hex(num)[2:].upper(),
        "speech": f"Recognized number {num}."
    }

def process_image_and_predict(image_pil, noise_level=0.0):
    """Segments digits, runs model, and extracts layer activations."""
    if noise_level > 0.0:
        arr = np.array(image_pil, dtype='float32')
        noise = np.random.normal(0, noise_level * 255.0, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        image_pil = Image.fromarray(arr)

    detected = segment_and_preprocess(image_pil)
    if not detected:
        return {"digits": [], "sequence": "", "insights": {}}

    results = []
    digits_sequence = []

    for idx, item in enumerate(detected):
        tensor = item['tensor']
        bbox = item['bbox']
        crop_arr = item['crop_array']

        # Get activations
        acts = activation_model(tensor, training=False)
        conv1_act = acts[0].numpy()[0]
        conv2_act = acts[1].numpy()[0]
        conv3_act = acts[2].numpy()[0]
        fc1_act = acts[3].numpy()[0]
        output_probs = acts[4].numpy()[0]

        pred_digit = int(np.argmax(output_probs))
        confidence = float(output_probs[pred_digit]) * 100.0
        digits_sequence.append(pred_digit)

        def sample_feature_maps(act_tensor, max_maps=16):
            maps = []
            num_channels = min(act_tensor.shape[-1], max_maps)
            for c in range(num_channels):
                f_map = act_tensor[:, :, c]
                f_min, f_max = f_map.min(), f_map.max()
                if f_max > f_min:
                    f_map = (f_map - f_min) / (f_max - f_min)
                else:
                    f_map = np.zeros_like(f_map)
                maps.append(array_to_base64(f_map))
            return maps

        conv1_maps = sample_feature_maps(conv1_act, 16)
        conv2_maps = sample_feature_maps(conv2_act, 16)
        conv3_maps = sample_feature_maps(conv3_act, 16)

        results.append({
            "index": idx,
            "digit": pred_digit,
            "confidence": round(confidence, 2),
            "probabilities": [round(float(p), 4) for p in output_probs],
            "bbox": {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]},
            "input_28x28": array_to_base64(crop_arr),
            "activations": {
                "conv1": conv1_maps,
                "conv2": conv2_maps,
                "conv3": conv3_maps,
                "fc1": [round(float(v), 3) for v in fc1_act[:32]]
            }
        })

    sequence_str = "".join(map(str, digits_sequence))
    insights = analyze_number(digits_sequence)

    return {
        "digits": results,
        "sequence": sequence_str,
        "insights": insights
    }

class CNNRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS for file:// access and prevent browser caching
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(body)
                img_data_url = data.get('image', '')
                noise = float(data.get('noise', 0.0))

                if ',' in img_data_url:
                    _, b64_str = img_data_url.split(',', 1)
                else:
                    b64_str = img_data_url

                raw_bytes = base64.b64decode(b64_str)
                img_pil = Image.open(io.BytesIO(raw_bytes))

                res = process_image_and_predict(img_pil, noise_level=noise)

                response_body = json.dumps(res).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(response_body)
            except Exception as e:
                err_body = json.dumps({"error": str(e)}).encode('utf-8')
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(err_body)
        else:
            self.send_error(404, "Endpoint Not Found")

    def do_GET(self):
        if self.path == '/api/model_info':
            info = {
                "name": "MNIST 3-Conv 2-FC Convolutional Neural Network",
                "status": "online",
                "total_params": 93322,
                "layers": [
                    {"name": "conv_layer_1", "type": "Conv2D", "filters": 32, "kernel": "3x3", "params": 320},
                    {"name": "maxpool_1", "type": "MaxPooling2D", "pool": "2x2", "params": 0},
                    {"name": "conv_layer_2", "type": "Conv2D", "filters": 64, "kernel": "3x3", "params": 18496},
                    {"name": "maxpool_2", "type": "MaxPooling2D", "pool": "2x2", "params": 0},
                    {"name": "conv_layer_3", "type": "Conv2D", "filters": 64, "kernel": "3x3", "params": 36928},
                    {"name": "flatten", "type": "Flatten", "output_dim": 576, "params": 0},
                    {"name": "fc_layer_1", "type": "Dense", "units": 64, "activation": "ReLU", "params": 36928},
                    {"name": "dropout", "type": "Dropout", "rate": 0.25, "params": 0},
                    {"name": "fc_layer_2_output", "type": "Dense", "units": 10, "activation": "Softmax", "params": 650}
                ],
                "test_accuracy": 98.90,
                "validation_accuracy": 98.32,
                "test_loss": 0.0373
            }
            body = json.dumps(info).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(body)
        else:
            if self.path == '/' or self.path == '':
                self.path = '/index.html'
            super().do_GET()

def run_server():
    # Set current directory to where app.py is located
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    socketserver.TCPServer.allow_reuse_address = True

    # Try binding to port, with fallback if occupied
    port = START_PORT
    server = None
    for attempt in range(10):
        try:
            server = socketserver.TCPServer(("", port), CNNRequestHandler)
            break
        except OSError:
            print(f"⚠️ Port {port} is occupied, trying port {port + 1}...")
            port += 1

    if not server:
        print(f"❌ Error: Could not bind to any port between {START_PORT} and {port}.")
        sys.exit(1)

    url = f"http://127.0.0.1:{port}"
    print(f"\n=======================================================")
    print(f"🚀 MNIST AI Studio is LIVE!")
    print(f"👉 Open in your browser: {url}")
    print(f"=======================================================\n")

    # Save running port to a small file so index.html or scripts know the exact port
    with open("current_port.txt", "w") as f:
        f.write(str(port))

    with server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == '__main__':
    run_server()
