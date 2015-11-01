import cv2
import sys
import numpy as np
import os

def main():
  width=10
  height=10
  
  fileSuffix=".mp4"
  videoDir = raw_input("Enter the video file directory:\n")
  videoFileName = raw_input("Enter the video file name:\n")
  xMin=int(raw_input("Enter the x coordiante:\n"))
  yMin=int(raw_input("Enter the y coordiante:\n"))
  optionNumber=raw_input("Enter the option number:\n")

  fullPath = r'{0}\{1}'.format(videoDir,videoFileName+fileSuffix) 
  outputFileName=r'{0}_{1}.tpc'.format(videoFileName,optionNumber)

  frames = extract_video_portion(fullPath,xMin,yMin,width,height)

  if frames != None:
    outFile = open( outputFileName, 'w' )
    error=0
    if optionNumber=='1':
      error=temporalPredictiveCodingOption1(frames,outFile,width,height) 
      print "Output saved to {0}".format(outputFileName)     
    elif optionNumber=='2':
      error=temporalPredictiveCodingOption2(frames,outFile,width,height)
      print "Output saved to {0}".format(outputFileName)
    elif optionNumber=='3':
      error=temporalPredictiveCodingOption3(frames,outFile,width,height)
      print "Output saved to {0}".format(outputFileName)
    elif optionNumber=='4':
      error=temporalPredictiveCodingOption4(frames,outFile,width,height)
      print "Output saved to {0}".format(outputFileName)
    elif optionNumber=='5':
      error=temporalPredictiveCodingOption5(frames,outFile,width,height)
      print "Output saved to {0}".format(outputFileName)
    else: 
      print "Input not valid"
      quit()
    # We send all original values so no error
    print "Total absolute prediction error is {0}".format(error)
    outFile.flush()
    outFile.close()
  else:
    print "Some error while reading video file"

def extract_video_portion(fullPath,xMin,yMin,width,height):
  frames = None

  xMax = xMin+width
  yMax = yMin+height
  cap = cv2.VideoCapture(fullPath)
  if cap.isOpened == None:
    return frames
  while cap.isOpened:
    success, img = cap.read()
    if success == True :
      cropImg = img[yMin:yMax,xMin:xMax]
      yuvImage = cv2.cvtColor(cropImg, cv2.COLOR_BGR2YUV)
      
      y,u,v=cv2.split(yuvImage)
      if frames == None:
        frames=np.array([y])
      else:
        frames=np.concatenate((frames,np.array([y])),axis=0)
    else:
      cap.release() 
      break
  cv2.destroyAllWindows()
  return frames
  

def temporalPredictiveCodingOption1(frames,outFile,width,height):
  frameCount=len(frames)  
  error = 0;
  for i in range(0,width):
    for j in range(0,height):
      outFile.write(" ".join(map(str,(frames[range(0,int(frameCount)),[i],[j]]).tolist()))+"\n")
  return error

def temporalPredictiveCodingOption2(frames,outFile,width,height):
  frameCount=len(frames)
  pcSignal=[0]*frameCount
  error = 0;

  for i in range(0,width):
    for j in range(0,height):
      signal = (frames[range(0,int(frameCount)),[i],[j]]).tolist()
      for k in range(0,frameCount):
        if k==0:
          pcSignal[k] = signal[0]
        else:
          pcSignal[k] = signal[k] - signal[k-1]
      outFile.write(" ".join(map(str,pcSignal))+"\n")
      error = error + sum([abs(p) for p in pcSignal])
      #error = error + sum(pcSignal)

  # subract intial signal value from error as we are sending original value
  return abs(error-signal[0])

def temporalPredictiveCodingOption3(frames,outFile,width,height):
  frameCount=len(frames)
  pcSignal=[0]*frameCount
  error = 0;

  for i in range(0,width):
    for j in range(0,height):

      signal = (frames[range(0,int(frameCount)),[i],[j]]).tolist()
      for k in range(0,frameCount):
        if k<=1:
          pcSignal[k] = signal[k]
        else:
          pcSignal[k] = signal[k] - signal[k-1]/float(2) - signal[k-2]/float(2)
      outFile.write(" ".join(map(str,pcSignal))+"\n")
      error = error + sum([abs(p) for p in pcSignal])
      #error = error + sum(pcSignal)

  # subract intial 2 signal values from error as we are sending original value
  return abs(error-signal[0]-signal[1])

def temporalPredictiveCodingOption4(frames,outFile,width,height):
  frameCount=len(frames)
  pcSignal=[0]*frameCount
  error = 0;
  alpha1 = 0.5
  alpha2 = 0.5

  for i in range(0,width):
    for j in range(0,height):

      signal = (frames[range(0,int(frameCount)),[i],[j]]).tolist()
      for k in range(0,frameCount):
        if k<=2:
          pcSignal[k] = signal[k]
        else:
          k1k2Diff = signal[k-1] - signal[k-2]
          if k1k2Diff == 0:
            alpha1=0.5
            alpha2=0.5
          else:
            alpha1 = (signal[k] - signal[k-2])/float(k1k2Diff)
            alpha2 = (signal[k-1] - signal[k])/float(k1k2Diff)
          
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
          
          pcSignal[k] = signal[k] - (alpha1*signal[k-1]  + alpha2*signal[k-2])
      outFile.write(" ".join(map(str,pcSignal))+"\n")
      error = error + sum([abs(p) for p in pcSignal])
      #error = error + sum(pcSignal)

  # subract intial 3 signal values from error as we are sending original value
  return abs(error-signal[0]-signal[1]-signal[2])

def temporalPredictiveCodingOption5(frames,outFile,width,height):
  frameCount=len(frames)
  pcSignal=[0]*frameCount

  alpha1 = 0.5
  alpha2 = 0.5
  errors=[]
  for p in np.arange(0,1.1,0.1):
    alpha1=p
    alpha2=1-p
    error=0
    for i in range(0,width):
      for j in range(0,height):
        signal = (frames[range(0,int(frameCount)),[i],[j]]).tolist()        
        for k in range(0,frameCount):
          if k<=2:
            pcSignal[k] = signal[k]
          else:
            pcSignal[k] = signal[k] - (alpha1*signal[k-1]  + alpha2*signal[k-2])
        error = error + sum([abs(p) for p in pcSignal])
        #error = error + sum(pcSignal)         
    errors.append(abs(error-signal[0]-signal[1]-signal[2]))
  predictorAlpha1=errors.index(min(errors))/float(10)
  predictorAlpha2=(10-errors.index(min(errors)))/float(10)
  print "predictor alpha1 %d" %  predictorAlpha1    
  print "predictor alpha2 %d" %  predictorAlpha2      

  error=0
  for i in range(0,width):
    for j in range(0,height):
      signal = (frames[range(0,int(frameCount)),[i],[j]]).tolist()        
      for k in range(0,frameCount):
        if k<=2:
          pcSignal[k] = signal[k]
        else:
          pcSignal[k] = signal[k] - (predictorAlpha1*signal[k-1]  + predictorAlpha2*signal[k-2])
      outFile.write(" ".join(map(str,pcSignal))+"\n")
      error = error + sum([abs(p) for p in pcSignal])
        #error = error + sum(pcSignal)         
  
  # subract intial 3 signal values from error as we are sending original value
  return abs(error-signal[0]-signal[1]-signal[2])

main()
