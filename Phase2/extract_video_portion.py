import cv2
import sys
import numpy as np
import os

def main():
  # video_dir=r'F:\ASU_Projects\MIS\mis\Phase1\sampleDataP1'
  # video_file_name = '1.mp4'
  # xMin=700
  # yMin=400
  width=10
  height=10
  option_number='4'
  
  video_dir = raw_input("Enter the video file directory:\n")
  video_file_name = raw_input("Enter the video file name:\n")
  xMin=int(raw_input("Enter the x coordiante:\n"))
  yMin=int(raw_input("Enter the y coordiante:\n"))
  option_number=raw_input("Enter the option number:\n")

  full_path = r'{0}\{1}'.format(video_dir,video_file_name) 
  output_video_file_name='extracted_'+video_file_name
  output_video_full_path=r'{0}\{1}'.format(video_dir,output_video_file_name)
  output_video_full_path='output.mov'
  
  output_file_name=r'{0}_{1}.tpc'.format(video_file_name,option_number)
  #extract_video_portion(full_path,output_video_full_path,xMin,yMin,width,height)

  frames = extract_video_portion(full_path,output_video_full_path,xMin,yMin,width,height)

  if frames != None:
    if option_number=='1':
      temporalPredictiveCodingOption1(frames,output_file_name,width,height) 
      print "Output saved to {0}".format(output_file_name)     
    elif option_number=='2':
      temporalPredictiveCodingOption2(frames,output_file_name,width,height)
      print "Output saved to {0}".format(output_file_name)
    elif option_number=='3':
      temporalPredictiveCodingOption3(frames,output_file_name,width,height)
      print "Output saved to {0}".format(output_file_name)
    elif option_number=='4':
      temporalPredictiveCodingOption4(frames,output_file_name,width,height)
      print "Output saved to {0}".format(output_file_name)
    else: 
      print "Input not valid"
      quit()
  else:
    print "Some error while reading video file"

def extract_video_portion(full_path,output_video_full_path,xMin,yMin,width,height):
  # Constants for the crop size
  frames = None

  xMax = xMin+width
  yMax = yMin+height
  cap = cv2.VideoCapture(full_path)
  if cap.isOpened == None:
    return frames
  
  # If you face any errors with these variable check opencv github and find the correspongin int values and replace it
  frameRate = cap.get(cv2.CAP_PROP_FPS)
  frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
  #signalList = [[[0]*frameCount]*width]*height
  #signal=[[0]*frameCount]*width*height
  #fourCharacterCode = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
  #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  #print fourcc
  #print fourCharacterCode
  #out = cv2.VideoWriter(output_video_full_path, fourCharacterCode, frameRate,(width,height))
  while cap.isOpened:
    success, img = cap.read()
    if success == True :
      cropImg = img[yMin:yMax,xMin:xMax]
      # cv2.imshow('video',img)
      # cv2.imshow('cropped video',cropImg)
      #out.write(cropImg)      
      yuvImage = cv2.cvtColor(cropImg, cv2.COLOR_BGR2YUV)
      y,u,v=cv2.split(yuvImage)
      #signals.append(y)
      if frames == None:
        frames=np.array([y])
      else:
        frames=np.concatenate((frames,np.array([y])),axis=0)
    else:
      cap.release() 
      break  
    # print frames.shape   
    # q = cv2.waitKey(1)
    # if 'q' == chr(q & 255):
    #   break 
  #out.release()
  cv2.destroyAllWindows()
  #print frames[range(0,int(frameCount)),[0],[1]]
  return frames
  

def temporalPredictiveCodingOption1(frames,output_file_name,width,height):
  frameCount=len(frames)
  current_working_dir = os.getcwd()
  outfile = open( output_file_name, 'w' )  
  error = 0;
  for i in range(0,width):
    for j in range(0,height):

      outfile.write(" ".join(map(str,(frames[range(0,int(frameCount)),[i],[j]]).tolist()))+"\n")
  # We send all original values so no error
  print "Total absolute prediction error is {0}".format(error)
  outfile.flush()
  outfile.close()

def temporalPredictiveCodingOption2(frames,output_file_name,width,height):
  frameCount=len(frames)
  pcSignal=[0]*frameCount
  current_working_dir = os.getcwd()
  outfile = open( output_file_name, 'w' )  
  error = 0;

  for i in range(0,width):
    for j in range(0,height):

      signal = (frames[range(0,int(frameCount)),[i],[j]]).tolist()
      for k in range(0,frameCount):
        if k==0:
          pcSignal[k] = signal[0]
        else:
          pcSignal[k] = signal[k] - signal[k-1]
      outfile.write(" ".join(map(str,pcSignal))+"\n")
      error = error + sum([abs(p) for p in pcSignal])

  # subract intial signal value from error as we are sending original value
  print "Total absolute prediction error is {0}".format(error-signal[0])
  outfile.flush()
  outfile.close()

def temporalPredictiveCodingOption3(frames,output_file_name,width,height):
  frameCount=len(frames)
  pcSignal=[0]*frameCount
  current_working_dir = os.getcwd()
  outfile = open( output_file_name, 'w' )  
  error = 0;

  for i in range(0,width):
    for j in range(0,height):

      signal = (frames[range(0,int(frameCount)),[i],[j]]).tolist()
      for k in range(0,frameCount):
        if k<=1:
          pcSignal[k] = signal[k]
        else:
          pcSignal[k] = signal[k] - signal[k-1]/float(2) - signal[k-2]/float(2)
      outfile.write(" ".join(map(str,pcSignal))+"\n")
      error = error + sum([abs(p) for p in pcSignal])

  # subract intial 2 signal values from error as we are sending original value
  print "Total absolute prediction error is {0}".format(error-signal[0]-signal[1])
  outfile.flush()
  outfile.close()

def temporalPredictiveCodingOption4(frames,output_file_name,width,height):
  frameCount=len(frames)
  pcSignal=[0]*frameCount
  current_working_dir = os.getcwd()
  outfile = open( output_file_name, 'w' )  
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
            k1k2Diff = 0.000001
          alpha1 = (signal[k] - signal[k-2])/float(k1k2Diff)
          alpha2 = (signal[k-1] - signal[k])/float(k1k2Diff)
          pcSignal[k] = signal[k] - (alpha1*signal[k-1]  + alpha2*signal[k-2])
      outfile.write(" ".join(map(str,pcSignal))+"\n")
      error = error + sum([abs(p) for p in pcSignal])

  # subract intial 3 signal values from error as we are sending original value
  print "Total absolute prediction error is {0}".format(error-signal[0]-signal[1]-signal[2])
  outfile.flush()
  outfile.close()

main()
