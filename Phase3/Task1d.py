__author__ = 'Nagarjuna'
import cv2
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import collections
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
  #videoDir = raw_input("Enter the video file directory:\n")
  #videoFileName = raw_input("Enter the video file name:\n")
  #numOfBits=int(raw_input("Enter number of bits:\n"))
  numOfBits=8
  videoDir=r'F:\ASU_Projects\MIS\mis\Phase1\sampleDataP1'
  videoFileName='1'
  fullPath = '{0}{2}{1}'.format(videoDir,videoFileName+fileSuffix,slash)
  outFileName='{0}_diff_{1}.dhc'.format(videoFileName,numOfBits)
  extract_video_portion(fullPath,width,height,numOfBits,outFileName)


def quantize(yChannel,numOfBits,frameId,blocksCoordinate,outfile):

    numOfPartitions = 2 ** numOfBits
    signal= yChannel.tolist()[0]
    #signal=[1,2,3,4,5,6,7,8,9]
    maxValue = max(signal)
    minValue = min(signal)
    partitionSize = (maxValue - minValue)/float(numOfPartitions)
    partitions=None
    if partitionSize != 0 :
      partitions = np.arange(minValue, maxValue+partitionSize, partitionSize)
    else :
      partitions = np.ones(numOfPartitions+1) * minValue

    binIndexes=np.digitize(np.array(signal),partitions)
    representative=[]
    for value in range(1,len(partitions)):
      representative.append(partitions[value-1]/float(2)+partitions[value]/float(2))

    quantized=[]

    for i in binIndexes.tolist():
      index=0
      if i-1 >= len(representative):
        index= i-2
      else:
        index=i-1
      quantized.append(int(representative[index]))

    frequency_list=collections.Counter(quantized)

    for idx,value in enumerate(frequency_list.keys()):
      freq_val = frequency_list[value]
      outfile.write("{0},{1},{2},{3}\n".format(frameId,blocksCoordinate,value,freq_val))



def extract_video_portion(fullPath,width,height,numOfBits,outFileName):
  frames = None
  frameWidth=None
  frameHeight=None
  outfile = open( outFileName,'w' )

  cap = cv2.VideoCapture(fullPath)

  if cap.isOpened == None:
    return frames
  if cv2.__version__=='3.0.0':
    frameWidth=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  else:
    frameWidth=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    frameHeight=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

  frameId=0
  count=0
  frames={}
  prevFrame=None
  while cap.isOpened:
    success, img = cap.read()
    if success == True :
      yuvImage = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
      y,u,v=cv2.split(yuvImage)
      blocksCoordinates={}
      if frameId==0:
        prevFrame=y
        frameId=frameId+1
        continue
      else:
        for i in range(0,frameWidth,width):
          for j in range(0,frameHeight,height):
            yChannel=y[j:j+height,i:i+width]
            prevFrameYChannel=prevFrame[j:j+height,i:i+width]
            count+=1
            diffYChannel=cv2.absdiff(yChannel,prevFrameYChannel)
            blocksCoordinate='{0},{1}'.format(i,j)
            flatYChannel = np.reshape(diffYChannel,(1,width*height))
            quantize(flatYChannel,numOfBits,frameId-1,blocksCoordinate,outfile)

      frameId=frameId+1
      outfile.flush()
    else:
      break
  outfile.close()

main()