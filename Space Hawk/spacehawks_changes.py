import zmq
import cv2
import numpy as np
import time
import os
import struct
import json
from pycoral.adapters.common import input_size
from pycoral.adapters.classify import get_classes
from pycoral.adapters.detect import BBox
from pycoral.adapters.detect import get_objects
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
import subprocess
# Define a class to deserialize data received via ZeroMQ
class OBC_TlmData:
    def __init__(self, data):
        unpack_format = 'b'*4 + 'H'*2 + 'h'*2 + 'f' + 'i'*4 + 'f'*13 + 'Ii' + 'b'*2 + 'H'*3 + 'I'
        expected_size = struct.calcsize(unpack_format)
        
        if len(data) != expected_size:
            print(f"Received data size: {len(data)}, expected: {expected_size}")
            raise ValueError(f"Expected data of size {expected_size} bytes, but received {len(data)} bytes.")
        
        (
            self.active_sensor_count, self.mission_mode, self.voltage_5v, self.voltage_3v,
            self.current_5v, self.current_3v, self.pi_temperature, self.board_temperature,
            self.al_lux,
            self.red_lux, self.green_lux, self.blue_lux, self.ir_lux,
            self.mag_uT_x, self.mag_uT_y, self.mag_uT_z,
            self.gyro_dps_x, self.gyro_dps_y, self.gyro_dps_z,
            self.accel_ms2_x, self.accel_ms2_y, self.accel_ms2_z,
            self.uv_a, self.uv_b, self.uv_c, self.uv_temp,
            self.ss_lux, self.ss_temperature,
            self.sc_voltage, self.padding, self.sc_ckt_resistance,
            self.sc_current, self.sc_power,
            self.timeepoch
        ) = struct.unpack(unpack_format, data)

# Function to read and preprocess images for inference
def read_inference_image(image, img_size=(256, 256), normalize=False):
    try:
        if image is None:
            raise ValueError("Error: Failed to load image")

        # Calculate aspect ratio
        h, w, _ = image.shape
        target_h, target_w = img_size

        if h > w:
            # Crop height
            start_h = (h - w) // 2
            image = image[start_h:start_h + w, :, :]
        else:
            # Crop width
            start_w = (w - h) // 2
            image = image[:, start_w:start_w + h, :]

        # Resize to target size
        image = cv2.resize(image, (target_w, target_h))

        if normalize:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image = image.astype(np.float32) / 255.0  # Normalize the images

        return image
    except Exception as e:
        print(f"Exception in reading image: {e}")
        return None

# Function to perform inference on a single image
def inference(interpreter, img, output_path, img_size, index):
    # Save the cropped original image before inference
    original_image_path = os.path.join(output_path, "originalCropped")
    if not os.path.exists(original_image_path):
        os.makedirs(original_image_path)

    cv2.imwrite(os.path.join(original_image_path, f"original_image_{index}.jpg"), img)

    # Preprocess the input image for inference
    img_infer = read_inference_image(img, img_size, normalize=True)
    if img_infer is None:
        print(f"Skipping inference for image index {index} due to preprocessing error.")
        return

    img_infer_expanded = np.expand_dims(img_infer, axis=0)  # Add batch dimension

    # Set the input tensor
    common.set_input(interpreter, img_infer_expanded)

    # Perform inference
    interpreter.invoke()

    # Get the output tensor
    output_tensor = common.output_tensor(interpreter, 0)

    # Process the output tensor
    output_image = np.squeeze(output_tensor)  # Assuming single output tensor
    output_image = (output_image * 255).astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # Create output directory for inference results if it does not exist
    result_image_path = os.path.join(output_path, "inferenceResult")
    if not os.path.exists(result_image_path):
        os.makedirs(result_image_path)

    # Save the output image with index
    output_image_path = os.path.join(result_image_path, f"inference_result_{index}.jpg")
    cv2.imwrite(output_image_path, output_image)

    print(f"Inference result {index} saved at {output_image_path}")

# Function to capture an image using OpenCVi
def capture_image(camera_port=0):
    try:
        # Set the output file path
        output_file = "captured_image.jpg"

        # Capture the image using libcamera-still command
        command = [
            'libcamera-still',
            '-o', output_file,
            '--timeout', '1000',  # Timeout in milliseconds
            '--immediate',         # Take a picture immediately
            '--width', '640',      # Set the width
            '--height', '480'      # Set the height
        ]

        # Run the command
        subprocess.run(command, check=True)

        # Read the image back to numpy array
        image = np.fromfile(output_file, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        print("Image captured successfully.")
        return image
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Load configuration from JSON file
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    # Read configuration values
    ipc_address = config["ZeroMQ"]["ipc_address"]
    lux_min = config["LUX"]["lux_min"]
    lux_max = config["LUX"]["lux_max"]
    model_path = config["Model"]["model_path"]
    output_path = config["Inference"]["output_path"]
    img_size = config["Inference"]["img_size"]
    capture_interval = config.get("capture_interval", 10)  # Default to 10 seconds if not specified
    camera_port = config.get("camera_port", 0)

    print("Configuration loaded successfully.")
    print(f"IPC Address: {ipc_address}, LUX Range: [{lux_min}, {lux_max}], Model Path: {model_path}")

    # Set up ZeroMQ subscriber socket
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(ipc_address)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    print("ZeroMQ subscriber set up successfully.")

    # Load the model with the Edge TPU interpreter
    try:
        interpreter = make_interpreter(f"{model_path}")
        interpreter.allocate_tensors()
        print("Model loaded and tensors allocated successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)
    frame = capture_image(camera_port)
    inference(interpreter, frame, output_path, img_size, 0)
    index = 0
    
