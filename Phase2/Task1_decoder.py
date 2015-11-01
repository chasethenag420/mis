import cv2
import sys
import numpy as np
import os

def main():
  width=10
  height=10
  fileSuffix=".mp4"
  videoFileName=raw_input("Enter the video file name:\n")
  videoDir = raw_input("Enter the video file directory:\n")
  optionNumber=raw_input("Enter the option number:\n")
  fullPath = r'{0}\{1}'.format(videoDir,videoFileName+fileSuffix)
  inputFileName=r'{0}_{1}.tpc'.format(videoFileName,optionNumber)
  outputFileName=r'{0}_{1}_out.mov'.format(videoFileName,optionNumber)

  inFile = open( inputFileName ) 
  frames= None

  if inputFileName != None:
    if optionNumber=='1':
      frames=tpcDecodingOption1(inFile) 
    elif optionNumber=='2':
      frames=tpcDecodingOption2(inFile)
    elif optionNumber=='3':
      frames=tpcDecodingOption3(inFile)
    elif optionNumber=='4':
      frames=tpcDecodingOption4(inFile)
    elif optionNumber=='5':
      frames=tpcDecodingOption5(inFile)
    else: 
      print "Input not valid"
      quit()
  else:
    print "Some error while reading video file"

  if frames !=None:
    decodeVideo(frames,fullPath,width,height,outputFileName,inputFileName)

  inFile.flush()
  inFile.close()

# frames: each row represent pixels in a frame which will be reshaped to width and height
def decodeVideo(frames,fullPath,width,height,outputVideoFileName,inputFileName):
  frameSize=frames.shape
  frameRate=30
  fourcc=-1

  cap = cv2.VideoCapture(fullPath)
  if cap.isOpened:
    frameRate=cap.get(cv2.CAP_PROP_FPS)
    fourcc=cap.get(cv2.CAP_PROP_FOURCC)

  outputFileName=inputFileName+"decoded.txt"
  outfile = open( outputFileName, 'w' )
  #fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
  #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  #fourcc = cv2.VideoWriter_fourcc('I', 'Y', 'U', 'V')
  
  outVideoFile = cv2.VideoWriter(outputVideoFileName, int(fourcc), frameRate,(width,height))
  for x in range(0,frameSize[0]):    
      frame=np.array(np.array(frames[x][:]).reshape(width,height), dtype=np.uint8)
      cv2.imshow("Decoded Y channel",frame)
      outfile.write(" ".join(map(str,frames[x][:]))+"\n")
      outVideoFile.write(frame)
      c = cv2.waitKey(1)
      if 'q' == chr(c & 255):
        break     
  outVideoFile.release()    
  cv2.destroyAllWindows()
  print "Output saved to text "+outputFileName
  outfile.flush()
  outfile.close()

  
def tpcDecodingOption1(inFile):
   
  lines=None
  for line in inFile:
    if lines==None:
      lines=np.array(list(map(int,line.split())))
    else:
      lines=np.column_stack((lines,list(map(int,line.split()))))
  return lines
  

def tpcDecodingOption2(inFile):
  lines=None
  for line in inFile:
    encodedSignal = list(map(int,line.split()))
    decodedSignal=[]
    for index,value in enumerate(encodedSignal):
      if index==0:
        decodedSignal.append(value)
      else:
        decodedSignal.append(value+decodedSignal[index-1])

    if lines==None:     
      lines=np.array(decodedSignal)

    else:
      lines=np.column_stack((lines,decodedSignal))
  return lines  


def tpcDecodingOption3(inFile):
  lines=None
  for line in inFile:
    encodedSignal = list(map(float,line.split()))
    decodedSignal=[]
    for index,value in enumerate(encodedSignal):
      if index<=1:
        decodedSignal.append(int(value))
      else:
        decodedSignal.append(int(value+decodedSignal[index-1]/float(2) + decodedSignal[index-2]/float(2)))

    if lines==None:     
      lines=np.array(decodedSignal)

    else:
      lines=np.column_stack((lines,decodedSignal))
  return lines

def tpcDecodingOption4(inFile):
  lines=None
  alpha1=0.5
  alpha2=0.5
  for line in inFile:
    encodedSignal = list(map(float,line.split()))
    decodedSignal=[]
    for index,value in enumerate(encodedSignal):
      if index<=2:
        decodedSignal.append(value)
      else:
        k1k2Diff = decodedSignal[index-1] - decodedSignal[index-2]
        
        if k1k2Diff == 0:
          alpha1=0.5
          alpha2=0.5
        else:
          alpha1 = (value - decodedSignal[index-2])/float(k1k2Diff)
          alpha2 = (decodedSignal[index-1] - value)/float(k1k2Diff)

        if alpha1>1 and alpha2>1:
          alpha1=0.5
          alpha2=0.5
        elif alpha1<0 and alpha2<0:
          alpha1=0.5
          alpha2=0.5
        elif alpha1<0 or alpha2>1:
          alpha1=0.0
          alpha2=1.0
        elif alpha1 >1 or alpha2<0:
          alpha1=1.0
          alpha2=0.0
        
        decodedSignal.append(value+ alpha1*decodedSignal[index-1]  + alpha2*decodedSignal[index-2])

    if lines==None:     
      lines=np.array(decodedSignal)

    else:
      lines=np.column_stack((lines,decodedSignal))
  return lines
  
def tpcDecodingOption5(inFile):
  lines=None
  alpha1=float(raw_input("Enter aplha1 predictor: "))
  alpha2=float(raw_input("Enter aplha2 predictor: "))
  for line in inFile:
    encodedSignal = list(map(float,line.split()))
    decodedSignal=[]
    for index,value in enumerate(encodedSignal):
      if index<=2:
        decodedSignal.append(int(round(value)))
      else:
     
        decodedSignal.append(int(round(value+ alpha1*decodedSignal[index-1]  + alpha2*decodedSignal[index-2])))

    if lines==None:     
      lines=np.array(decodedSignal)

    else:
      lines=np.column_stack((lines,decodedSignal))
  return lines

main()
