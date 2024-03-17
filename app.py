import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tempfile
import time

# Function to load or save the model
def load_or_save_model():
    model_path = "esrgan_model"  # Path to save the model

    # Check if the model is already saved
    if os.path.exists(model_path):
        print("Model already exists, loading from", model_path)
        model = tf.saved_model.load(model_path)
    else:
        print("Model not found, downloading and saving the model...")
        model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")  # Load ESRGAN model
        tf.saved_model.save(model, model_path)  # Save the model to a local directory

    return model

# Function to preprocess video frame for model input
def preprocess_frame(frame):
    frame = cv2.resize(frame, (640, 480))  # Resize to SD resolution
    frame = tf.expand_dims(frame, 0)  # Add batch dimension
    frame = tf.cast(frame, tf.float32)  # Cast to float32
    return frame

# Function to postprocess video frame after model output
def postprocess_frame(frame):
    frame = tf.squeeze(frame)  # Remove batch dimension
    frame = tf.clip_by_value(frame, 0, 255)  # Clip pixel values
    frame = tf.cast(frame, tf.uint8)  # Cast to uint8
    return frame.numpy()

# Function to convert SD video to HD using diffusion model
def convert_video(input_video, model, temp_input):
    # Use tempfile to create a temporary file object for the uploaded video
    temp_input.flush()
    os.fsync(temp_input.fileno())

    input_video_path = temp_input.name  # Now you have the path to the temporary file

    input_video = cv2.VideoCapture(input_video_path)
    if not input_video.isOpened():
        st.error("Error: Unable to open input video file.")
        return

    fps = input_video.get(cv2.CAP_PROP_FPS)
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2  # Double the width for HD
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_video_path = temp_output.name
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (1280, 720))
    if not output_video.isOpened():
        st.error("Error: Unable to open output video file.")
        input_video.release()
        return

    frame_count = 0
    while True:
        ret, frame = input_video.read()
        if not ret:
            break

        frame_count += 1

        # Preprocess frame
        preprocessed_frame = preprocess_frame(frame)

        # Pass the preprocessed frame through the model
        sr_frame = model(preprocessed_frame)

        # Postprocess frame
        sr_frame = postprocess_frame(sr_frame)

        # Write the super-resolved frame to the output video file
        output_video.write(sr_frame)

    # Release video objects
    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()

    st.success(f"Video conversion completed. Total frames processed: {frame_count}")
    temp_input.close()
    return output_video_path

# Function to generate downloadable link for the converted video (optional)
def get_binary_file_downloader_html(bin_file, btn_text="Download"):
    """Generates HTML for a button that downloads a binary file."""
    with open(bin_file, "rb") as f:
        data = f.read()
        href = f"<a href='data:file/mp4;base64,{data}' download='{bin_file}'>{btn_text}</a>"
    return href

import time

# Streamlit app
def main():
    st.title("Video Super-Resolution App")
    st.write("This app converts SD videos to HD using a pre-trained model.")

    input_video = st.file_uploader("Upload an SD video file (.mp4)", type="mp4")

    if input_video:
        # Save the uploaded file to a temporary location
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(input_video.read())

        temp_input_path = temp_input.name

        st.video(temp_input_path)
        model = load_or_save_model()
        if st.button("Convert Video"):
            with st.spinner("Converting video..."):
                # Wait for the file to be fully uploaded
                time.sleep(2)  # Adjust this delay as needed
                output_video_path = convert_video(input_video, model, temp_input)
            if output_video_path is not None:
                st.success("Video converted successfully!")
                st.subheader("Download Converted Video")
                st.markdown(get_binary_file_downloader_html(output_video_path, "Download"), unsafe_allow_html=True)
            else:
                st.error("Video conversion failed.")


if __name__ == "__main__":
    main()



