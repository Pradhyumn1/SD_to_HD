import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

def load_or_save_model(model_path="esrgan_model"):
    """Loads the ESRGAN model from disk or downloads it if not found."""
    if os.path.exists(model_path):
        print("Model already exists, loading from", model_path)
        model = tf.saved_model.load(model_path)
    else:
        print("Model not found, downloading and saving the model...")
        model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
        tf.saved_model.save(model, model_path)
    return model

def preprocess_frame(frame):
    """Preprocesses a video frame for model input."""
    frame = cv2.resize(frame, (640, 480))  # Resize to SD resolution
    frame = tf.expand_dims(frame, 0)  # Add batch dimension
    frame = tf.cast(frame, tf.float32)  # Cast to float32
    return frame

def postprocess_frame(frame):
    """Postprocesses a video frame after model output."""
    frame = tf.squeeze(frame)  # Remove batch dimension
    frame = tf.clip_by_value(frame, 0, 255)  # Clip pixel values
    frame = tf.cast(frame, tf.uint8)  # Cast to uint8
    return frame.numpy()

def convert_video(video_bytes, model):
    """Processes a video using the provided model."""
    try:
        video_array = np.frombuffer(video_bytes, dtype=np.uint8)
        video_array = cv2.imdecode(video_array, cv2.IMREAD_COLOR)
        fps = 30  # Adjust as needed
        width = int(video_array.shape[1]) * 2
        height = int(video_array.shape[0]) * 2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = 'output_video.mp4'
        output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for frame in video_array:
            preprocessed_frame = preprocess_frame(frame)
            sr_frame = model(preprocessed_frame)
            sr_frame = postprocess_frame(sr_frame)
            output_video.write(sr_frame)

        output_video.release() ###
        
        
        cv2.destroyAllWindows()

        with open(output_video_path, 'rb') as f:
            processed_video_bytes = f.read()

        os.remove(output_video_path)

        return processed_video_bytes

    except Exception as e:
        print(f"Error processing video: {e}")
        return None
