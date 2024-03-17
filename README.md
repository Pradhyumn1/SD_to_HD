# SD to HD Video Conversion using ESRGAN

This project utilizes the Enhanced Super-Resolution Generative Adversarial Network (ESRGAN) to convert standard-definition (SD) videos to high-definition (HD) videos. It provides a simple web interface for users to upload their SD videos and obtain the corresponding HD versions.

## Installation

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/your-username/sd_to_hd.git
    ```

2. Navigate to the project directory:

    ```bash
    cd sd_to_hd
    ```

3. Create a virtual environment:

    ```bash
    python3 -m venv venv
    ```

4. Activate the virtual environment:

    ```bash
    source venv/bin/activate
    ```

5. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Flask application:

    ```bash
    python main.py
    ```

2. Open your web browser and go to `http://127.0.0.1:5000/`.

3. Upload your SD video file using the provided form.

4. Wait for the conversion process to complete.

5. Download the converted HD video.

## Project Structure

- `main.py`: Flask application that handles the web interface and video processing logic.
- `utils.py`: Helper functions for loading the ESRGAN model, preprocessing and postprocessing frames, and converting videos.
- `templates/`: Directory containing HTML templates for the web interface.
- `static/`: Directory for static files (e.g., CSS, JavaScript).

## Dependencies

- Flask: Web framework for Python.
# - Streamlit also resent in app.py file
- TensorFlow: Deep learning framework for model loading and inference.
- OpenCV (cv2): Library for image and video processing.
- TensorFlow Hub: Repository for reusable machine learning modules.

## Acknowledgments

- This project uses the ESRGAN model from TensorFlow Hub, developed by Captain Pool.

## Images

![Fist img](<Screenshot 2024-03-17 at 4.02.25 PM.png>)

![secod image after model run](<Screenshot 2024-03-17 at 4.02.03 PM.png>)