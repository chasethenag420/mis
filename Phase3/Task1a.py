__author__ = 'Nagarjuna'
import cv2
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from sys import platform as _platform

def main():
  width=8
  height=8
  if _platform == "linux" or _platform == "linux2":
    slash = '/'
  elif _platform == "darwin":
    slash = '/'
  elif _platform == "win32":
    slash = '\\'
  fileSuffix=".mp4"
  videoDir = raw_input("Enter the video file directory:\n")
  videoFileName = raw_input("Enter the video file name:\n")
  fullPath = '{0}{2}{1}'.format(videoDir,videoFileName+fileSuffix,slash)
  frames = extract_video_portion(fullPath,xMin,yMin,width,height)


def extract_video_portion(fullPath,width,height):
  frames = None
  frameWidth=None
  frameHeight=None



  cap = cv2.VideoCapture(fullPath)

  if cap.isOpened == None:
    return frames
  if cv2.__version__=='3.0.0':
    frameWidth=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  else:
    frameWidth=cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    frameHeight=cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

  frameId=0

  while cap.isOpened:
    success, img = cap.read()
    if success == True :
      xMin=0
      yMin=0
      #cropImg = img[yMin:yMax,xMin:xMax]
      yuvImage = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
      y,u,v=cv2.split(yuvImage)
      for i in range(0,frameWidth):
        for j in range(0,frameHeight):
          xMax = xMin+width
          yMax = yMin+height
          if xMax<=frameHeight && yMax<=frameWidth:
            y88 = y[yMin:yMax,xMin:xMax]
            print y88.flat
          else:
            y88 = y[yMin:frameWidth,xMin:frameHeight]
            print y88.flat
            c = cv2.waitKey(0)
            if 'q' == chr(c & 255):
              cv2.destroyAllWindows()

          print '{0},{1},{2},{3}'.format(frameId,xMin,yMin,)
          xMin=xMax
          yMin=yMax

      frameId=frameId+1

      if frames == None:
        frames=np.array([y])
      else:
        frames=np.concatenate((frames,np.array([y])),axis=0)
    else:
      cap.release()
      break
  cv2.destroyAllWindows()
  return frames