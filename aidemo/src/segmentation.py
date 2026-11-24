import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects, center_of_mass

def segment_and_preprocess(image_input):
    """
    Takes a PIL Image, filepath, or numpy array.
    Finds and isolates multiple handwritten digits, filters noise/speckles,
    sorts them left-to-right, and scales/centers each digit into standard 28x28 MNIST format.
    
    Returns:
        List of dicts:
        [{
            'tensor': np.ndarray (shape: (1, 28, 28, 1), normalized [0, 1]),
            'bbox': (x, y, w, h),
            'crop_array': np.ndarray (28, 28)
        }, ...]
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('L')
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('L')
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input).convert('L')
    else:
        raise ValueError("Unsupported image input type.")

    arr = np.array(img, dtype='float32')

    # Detect if background is light and invert if needed (MNIST has white digits on black background)
    if arr.mean() > 128:
        arr = 255.0 - arr

    # Clean background noise with a threshold
    bg_noise = np.percentile(arr, 20)
    threshold = max(25.0, bg_noise + 20.0)
    binary = arr > threshold

    # Connected component labeling
    labeled, num_features = label(binary)
    if num_features == 0:
        return []

    objects = find_objects(labeled)
    valid_objects = []
    
    max_h = 0
    max_w = 0
    max_area = 0

    for obj in objects:
        if obj is None:
            continue
        slice_y, slice_x = obj
        h = slice_y.stop - slice_y.start
        w = slice_x.stop - slice_x.start
        area = h * w
        if h > max_h:
            max_h = h
        if w > max_w:
            max_w = w
        if area > max_area:
            max_area = area
        valid_objects.append((obj, w, h, area))

    if not valid_objects or max_h < 8:
        return []

    # Adaptive noise threshold: only keep components that are at least 20% of max_h and 2% of max_area
    min_h = max(10, int(max_h * 0.20))
    min_area = max(50, int(max_area * 0.02))

    detected_digits = []

    for obj, w, h, area in valid_objects:
        if h < min_h or area < min_area:
            continue

        slice_y, slice_x = obj
        digit_crop = arr[slice_y, slice_x]

        # Resize preserving aspect ratio to fit into 20x20 box (standard MNIST specification)
        crop_pil = Image.fromarray(digit_crop.astype('uint8'))
        if w > h:
            new_w = 20
            new_h = max(1, int(round(h * 20.0 / w)))
        else:
            new_h = 20
            new_w = max(1, int(round(w * 20.0 / h)))
            
        resized_pil = crop_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized_arr = np.array(resized_pil, dtype='float32')

        # Create 28x28 canvas with centered digit
        canvas = np.zeros((28, 28), dtype='float32')
        pad_x = (28 - new_w) // 2
        pad_y = (28 - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_arr

        # Shift center of mass to center (14, 14) to perfectly match MNIST
        cy, cx = center_of_mass(canvas)
        if not (np.isnan(cy) or np.isnan(cx)):
            shift_y = int(round(14.0 - cy))
            shift_x = int(round(14.0 - cx))
            # Safe bounded translation
            shifted = np.zeros((28, 28), dtype='float32')
            src_y1 = max(0, -shift_y)
            src_y2 = min(28, 28 - shift_y)
            src_x1 = max(0, -shift_x)
            src_x2 = min(28, 28 - shift_x)
            
            dst_y1 = max(0, shift_y)
            dst_y2 = min(28, 28 + shift_y)
            dst_x1 = max(0, shift_x)
            dst_x2 = min(28, 28 + shift_x)
            
            h_overlap = min(src_y2 - src_y1, dst_y2 - dst_y1)
            w_overlap = min(src_x2 - src_x1, dst_x2 - dst_x1)
            if h_overlap > 0 and w_overlap > 0:
                shifted[dst_y1:dst_y1 + h_overlap, dst_x1:dst_x1 + w_overlap] = canvas[src_y1:src_y1 + h_overlap, src_x1:src_x1 + w_overlap]
                canvas = shifted

        # Normalize to [0, 1]
        canvas = canvas / 255.0
        tensor = canvas.reshape(1, 28, 28, 1)

        detected_digits.append({
            'tensor': tensor,
            'crop_array': canvas,
            'bbox': (int(slice_x.start), int(slice_y.start), int(w), int(h)),
            'x_start': slice_x.start
        })

    # Sort detected digits strictly from left to right (horizontal reading order)
    detected_digits.sort(key=lambda item: item['x_start'])

    return detected_digits
