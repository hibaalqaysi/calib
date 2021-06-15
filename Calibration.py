# Extract clibration matrix and distortion coefficients from Left and right videos
# Programm will not work if program detects more then 150 patterns.

import cv2
import os
import json
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import myClassCalib as mc
import math

def Calibration(filename, yamlFiles):
    __mt = mc.Calib(filename)
    #__mt.GetFilesFromDir(Enable = True)
    print("done")


def main():
    try:
        #video = ['cam1_20180623_144000crop1.mp4', 'cam2_20180623_144506crop1.mp4']
        yamlFiles = ["calibCyrcleC1.yaml", "calibCyrcleR.yaml"]
        filename ="C:/Working/VideoForCalibration/cam1_20180623_144000crop1.mp4"
        Calibration(filename, yamlFiles)

        return 0
    except:
        return 1


if __name__ == '__main__':
    main()