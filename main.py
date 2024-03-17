from flask import Flask, request, send_file, render_template, abort
from io import BytesIO
import os
import io
import utils

app = Flask(__name__)

# Load the ESRGAN model
model = utils.load_or_save_model()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_video():
    input_video = request.files['inputVideo']
    if not input_video:
        abort(400, 'Missing video file')

    # Read video content into a byte array
    video_bytes = input_video.read()

    # Process the video using functions from utils.py
    output_video_bytes = utils.convert_video(video_bytes, model)

    if not output_video_bytes:
        abort(500, 'Video processing failed')

    return send_file(
        io.BytesIO(output_video_bytes),
        mimetype='video/mp4',
        as_attachment=True,
        download_name='output_video.mp4'
    )

if __name__ == '__main__':
    app.run(debug=True)
