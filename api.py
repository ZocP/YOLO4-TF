from __future__ import division, print_function

import os
import detect
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
load_model = "./checkpoints/yolov4-416"
saved_model_loaded = tf.saved_model.load(load_model, tags=[tag_constants.SERVING])
print('Model loaded. Check http://127.0.0.1:5000/')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        #file_path
        get_detected_object = detect.glass_detector(file_path, saved_model_loaded)
        #pil image to base 64
        return get_detected_object
    return None


if __name__ == '__main__':
    app.run(debug=True)