from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import os
import numpy as np
from skimage.draw import polygon
import json
import matplotlib.pyplot as plt
import time

class media_processor():
    
    def __init__(self, predictor_file,
                 videofile, duration, 
                 fps = 30):
        self.mediafile = videofile
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_file)
        self.duration = duration
        self.fps = fps
        self.framelimit = self.fps * self.duration
        
        self.results = []
        self.currentframe = 0        
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontscale = 0.5
        self.fontcolor = (0, 0, 255)
        self.BREAK_POINTS = [16, 21, 26, 
                             35, 41, 47, ]
        self.CLOSURE = {16: 15, 21: 20, 
                        26: 25, 35: 30, 
                        41: 36, 47: 42}
        self.FACE_PARTS = {
            "cheek": [0, 17], 
            "l_eyebrow": [17, 21], 
            "r_eyebrow": [22, 26],
            "l_eye": [36, 42],
            "r_eye": [42, 48],
            "nose_bone": [27, 30],
            "nose_base": [31, 35],
            "lips": [48, 67]  
        }        
        # FACE_BOUNDARY = list(range(0, 16)) + [78, 76, 77] + list(range(69, 76)) + [79]
        self.FACE_BOUNDARY = list(range(0, 16)) + [78, 74, 79, 73, 
                                                   72, 80, 71, 70, 
                                                   69, 68, 76, 75, 
                                                   77]
        self.ROI = {
            "lface_roi" : [48, 31, 27, 39, 
                           40, 41, 36],
            "rface_roi" : [54, 35, 27, 42, 
                           47, 45, 46]
        }

    def datadir(self):
        try:
            # creating a folder named data 
            if not os.path.exists('data'): 
                os.makedirs('data') 
        # if not created then raise error 
        except OSError: 
            print ('Error: Creating directory of data') 

    def mark_roi(self, rects, 
                 gray, frame):
        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            pred_start = time.time()
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            adjust_start = time.time()
            # slight adjustment to the shape lip corner coordinates
            shape[31][0] -= 20
            shape[35][0] += 20
            
            shape[48][0] -= 40  
            shape[54][0] += 40
            
            # get the forehead coords
            xbl = shape[19][0]
            ybl = shape[19][1] - 10
            xbr = shape[24][0]
            ybr = shape[24][1] - 10
            
            xul = xbl
            yul = ybl - 50
            xur = xbr
            yur = yul

            y1, x1 = polygon(np.array([ybl, ybr, yur, yul]), 
                             np.array([xbl, xbr, xur, xul]))
            y2, x2 = polygon(shape[self.ROI["lface_roi"]][:, 1], 
                             shape[self.ROI["lface_roi"]][:, 0])
            y3, x3 = polygon(shape[self.ROI["rface_roi"]][:, 1], 
                             shape[self.ROI["rface_roi"]][:, 0])
            
            crop_start = time.time()
            # self.crop_and_save((np.r_[y1, y2, y3], 
            #                     np.r_[x1, x2, x3]), 
            #                     frame, name=name)
            self.crop_and_save((np.r_[y1, y2, y3], 
                                np.r_[x1, x2, x3]), 
                                frame)
            
            # increasing counter so that it will 
            # show how many frames are created 
            crop_end = time.time()
            self.currentframe += 1

    def crop_and_save(self, points, image, 
                      savedir=None, prefix=None, 
                      pCounter=0, name=None):
        Y, X = points
        cropped_img = np.zeros(image.shape, dtype=np.uint8)
        cropped_img[points] = image[points]
        self.results.append((np.sum(image[points]) // points[1].shape)[0])
        # if name == None:
        #     name = savedir + prefix + str(pCounter) + '.jpg'
        # cv2.imwrite(name, cropped_img)

    def start_frame_capture(self):
        print("Start Frame Capture")
        vidcap = cv2.VideoCapture(self.mediafile)
        success, frame = vidcap.read()
        frames_count = 0
        while success:
            frames_count += 1
            # capture data from first 2 seconds frames
            if frames_count > self.framelimit:
                break
            # grab the frame from the threaded video stream, resize it to
            # have a maximum width of 400 pixels, and convert it to
            # grayscale
            frame = imutils.resize(frame, width=1300)
            frame = imutils.rotate(frame, 90)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = self.detector(gray, 0)
            if len(rects) > 0:
                self.mark_roi(rects, gray, frame)
            else:
                self.results.append(0)
            success, frame = vidcap.read()
        
    def get_bpm(self):
        print("Start Beats Per Minute Calculation")
        if np.where(np.array(self.results) != 0)[0].size / len(self.results) >= 0.7:
            L = self.framelimit
            nonzero_idx = np.where(np.array(self.results) != 0)
            nonzero_vals = np.array(self.results)[nonzero_idx]
            interpolated = np.interp(np.arange(L), nonzero_idx[0], nonzero_vals)
            interpolated = np.hamming(L) * interpolated
            interpolated = interpolated - np.mean(interpolated)
            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            fft = np.abs(raw)
            freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * freqs
            idx = np.where((freqs > 50) & (freqs < 180))
            pruned = fft[idx]
            phase = phase[idx]
            pfreq = freqs[idx]
            freqs = pfreq
            fft = pruned
            idx2 = np.argmax(pruned)
            # return the beats per minute for the frames
            print("Beats per minute detected: ", freqs[idx2])
            return freqs[idx2]
        else:
            print("Condition Not Satisfied")
            return 0

    def get_bpm_new(self):
        L = self.fps * self.duration
        if L == len(self.results):    
            interpolated = np.hamming(L) * self.results
            interpolated = interpolated - np.mean(interpolated)
            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            fft = np.abs(raw)
            freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * freqs
            idx = np.where((freqs > 50) & (freqs < 180))
            pruned = fft[idx]
            phase = phase[idx]
            pfreq = freqs[idx]
            freqs = pfreq
            fft = pruned
            idx2 = np.argmax(pruned)
            # return the beats per minute for the frames
            print("Beats per minute detected: ", freqs[idx2])
        else:
            return "Error !!"
        return freqs[idx2]
        