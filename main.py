from flask import Flask, request, Response, jsonify
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os
import cv2 
from processor import media_processor
import numpy as np

app = Flask(__name__)
predictor_file = os.path.join('models', 'shape_predictor_81_face_landmarks.dat')

# set some variables to be changed later
duration_to_consider = 1  # seconds
frames_per_second = 30  # camcorder frames capture rate

@app.route('/api/uploadVideo', methods=['POST'])
def uploadVideo():
    print(request.files)
    f = request.files['video']
    target = os.path.join('videos', 
                          secure_filename(f.filename))
    f.save(target)
    bpm = processmedia(predictor_file, target)
    bpm = np.round(bpm * 
            (1 + np.random.choice(np.arange(10)) / 100))
    return jsonify({"bpm": bpm})
    #return 'OK'

def processmedia(predictor, media):
    mp = media_processor(predictor, media, 
                         duration_to_consider, 
                         frames_per_second)
    mp.start_frame_capture()
    return mp.get_bpm()

if __name__ == "__main__":
    app.run(host="0.0.0.0", 
           port=5000, 
           debug=True)