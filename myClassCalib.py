# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:22:42 2018

@author: igofed
"""
import json
import cv2
import tkinter
import numpy as np
import matplotlib.pyplot as plt
import threading
import yaml

from  math  import  sin ,  cos ,  tan ,  pi
import os
import glob

class Calib(object):
    def __init__(self, filename):
        self.pSize = 3.45E-6
        self.SelectedImagesInC1 = ['26',  '51',  '64',  '77',  '87',  '89',  '457', '463',
                                   '491', '503', '517', '537', '573', '581', '591','693',
                                   '627', '641', '647', '649', '654', '661', '669']
        self.pattern_size = (9, 6)
        self.objpoints = []
        self.imgpoints = []
        self.calc_timestamps = [0.0]
        self.calibration = {}
        self.filename = filename
        self.fNum = 0
        self.PatNum = 0
        self.objpoints = []
        self.imgpoints = []
        self.mError = 0
        self.numGrid = []

        for i, filename in enumerate(os.listdir("C:\Working\VideoForCalibration\Cam2")):
            print(i, filename)

        self.VideoCap()
        self.findCorners(1)
        self.mean_error = 0
        #self.detector = self.BlobDetector()

    def BlobDetector(self):
        self.params = cv2.SimpleBlobDetector_Params()
        self.params.filterByArea = True
        self.params.minArea = 100
        self.params.maxArea = 20000
        self.params.filterByCircularity = True
        self.params.minCircularity = 0.2
        self.params.filterByInertia = True
        self.params.minInertiaRatio = 0.1
        # blob detection only works with "uint8" images.

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            print('SimpleBlobDetector')
            return cv2.SimpleBlobDetector(self.params)

        else:
            print('SimpleBlobDetector_create')
            return cv2.SimpleBlobDetector_create(self.params)

#    def imageRead(self):
#        data_path = os.path.join(self.img_dir, '*g')
#        files = glob.glob(data_path)
#        data = []




    def VideoCap(self):
        #print('------------')
        cap = cv2.VideoCapture(self.filename)
        if cap.isOpened() == False:
            print("Error opening video stream or file")
        else:
            print("Video :", self.filename, " is opened")
            n_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("nframes: ", n_frame, " fps: ", fps, " [ ", width, " x ", height, "]")
            print("-----------------------")
        return cap



    def findCorners(self, TypeOfPattern = 1):
        params = []
        def objParams(TypeOfPattern):
            print (TypeOfPattern)
            if TypeOfPattern == 1:
                param1 = np.zeros((4 * 11, 3), np.float32)
                param2 = np.mgrid[0:4,0:11].T.reshape(-1,2)
                print("Cyrcle")
            elif TypeOfPattern == 0:
                param1 = np.zeros((6 * 9, 3), np.float32)
                param2 = np.mgrid[0:9,0:6].T.reshape(-1,2)
                print("Checkerboard")
            else:
                param1 = np.zeros((4 * 11, 3), np.float32)
                param2 = np.mgrid[0:4,0:11].T.reshape(-1,2)
                print ("Type of pattern unrecognized")
            params.append(param1)
            params.append(param2)
        objParams(TypeOfPattern)
        self.objp = params[0]
        self.objp[:, :2] = params[1]
        objParams(TypeOfPattern)
        print('Corners finder')

    def calibrate(self):
        print ("start calibrateCamera")
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints,(self.width, self.height), None, None)
        print ("end calibrateCamera")
        self.error_in_frame = []
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            self.mean_error += error
            self.error_in_frame.append(error)
        self.mean_error =self.mean_error / len(self.objpoints)
        print("total error: ", self.mean_error)

        print (type(self.mtx),"mtx: ", self.mtx)
        print('dist', self.dist)

    def addTexttoFramePattern(self, frame, color):
        S = "Pattern detected: " +  str(int(self.PatNum))
        cv2.putText(frame, str(int(self.n_frame)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, 8)
        cv2.putText(frame, str(self.fNum), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, 8)
        cv2.putText(frame, S, (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, 8)

    def addTexttoFrame(self, frame, color):
        cv2.putText(frame, str(int(self.n_frame)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, 8)
        cv2.putText(frame, str(self.fNum), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, 8)

    def saveToFile(self, yamlFiles):
        f = cv2.FileStorage (yamlFiles, cv2.FILE_STORAGE_WRITE)
        f.write("mtx", self.mtx)
        f.write("dist", self.dist)
        f.write("rms", self.ret)
        f.write("mean_error", np.array(self.mean_error))
        f.write("frame", np.array(self.numGrid))
        f.write("error_in_frame", np.array(self.error_in_frame))
        f.write("rvecs", np.array(self.rvecs))
        f.write("tvecs", np.array(self.tvecs))

        print (np.array(self.numGrid))
        print (np.array(self.error_in_frame))

        #f.write("Frames", self.numGrid)
        #f.release()
        print ("Files saved")

