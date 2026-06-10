import cv2
import numpy as np

def prepare_canvas_image(canvas_image):
    gray = np.max(canvas_image[:, :, :3], axis=2).astype(np.uint8)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None, None
    x, y, w, h = cv2.boundingRect(coords)
    digit = thresh[y:y+h, x:x+w]
    height, width = digit.shape
    if height == 0 or width == 0:
        return None, None

    if height > width:
        new_height = 20
        new_width = max(1, int(width * 20 / height))
    else:
        new_width = 20
        new_height = max(1, int(height * 20 / width))

    digit = cv2.resize(digit, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = digit
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    processed = canvas.astype(np.float32) / 255.0
    processed = processed.reshape(1, 784)
    return processed, canvas