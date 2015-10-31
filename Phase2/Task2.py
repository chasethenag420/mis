import cv2
import sys
import numpy as np
import os

def main():
  video_dir=r'F:\ASU_Projects\MIS\mis\Phase1\sampleDataP1'
  video_file_name = '1.mp4'
  xMin=700
  yMin=400
  width=10
  height=10
  #option_number='1'
  
  '''
  video_dir = raw_input("Enter the video file directory:\n")
  video_file_name = raw_input("Enter the video file name:\n")
  xMin=int(raw_input("Enter the top-left x coordiante:\n"))
  yMin=int(raw_input("Enter the top-left y coordiante:\n"))
  '''

  # video_dir = "sampleDataP1"
  # video_file_name = "1.mp4"
  # xMin = 1
  # yMin = 1


  print "Select any one of the following: "
  print "Press 1 for No PC"
  print "Press 2 for Predictor A"
  print "Press 3 for Predictor B"
  print "Press 4 for Predictor C"
  print "Press 5 for Alpha-based Predictor"


  option_number=raw_input("Enter the option number:\n")

  full_path = r'{0}\{1}'.format(video_dir,video_file_name) 
  output_video_file_name='extracted_'+video_file_name
  output_video_full_path=r'{0}\{1}'.format(video_dir,output_video_file_name)
  output_video_full_path='output.mov'
  
  output_file_name=r'{0}_{1}.spc'.format(video_file_name,option_number)
  #extract_video_portion(full_path,output_video_full_path,xMin,yMin,width,height)

  frames = extract_video_portion(full_path,output_video_full_path,xMin,yMin,width,height)

  if frames != None:
    if option_number=='1':
      spatialPredictiveCodingOption1(frames,output_file_name,width,height) 
      print "Output saved to {0}".format(output_file_name)     
    elif option_number=='2':
      spatialPredictiveCodingOption2(frames,output_file_name,width,height)
      print "Output saved to {0}".format(output_file_name)
    elif option_number=='3':
      spatialPredictiveCodingOption3(frames,output_file_name,width,height)
      print "Output saved to {0}".format(output_file_name)
    elif option_number=='4':
      spatialPredictiveCodingOption4(frames,output_file_name,width,height)
      print "Output saved to {0}".format(output_file_name)
    elif option_number=='5':
      spatialPredictiveCodingOption5(frames,output_file_name,width,height)
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

def spatialPredictiveCodingOption1(frames,output_file_name,width,height):
  # No Predictive Coding
  frameCount = len(frames)
  current_working_dir = os.getcwd()
  outfile = open( output_file_name, 'w' )  
  error = 0;

  print frames 

  for k in range(0, 3):
    #outfile.write(str(frames[k]))
    #outfile.write("\n")
    print frames[k]
    print "\n"
    #print frames[k][0]
    #print "\n"
    #print frames[k][0][0] 


  for k in range(0, frameCount):
    for i in range (0, height):
      for j in range(0, width):
        outfile.write(str(frames[k][i][j]))
        outfile.write(" ")
    outfile.write("\n")

  # We send all original values so no error
  print "Total absolute prediction error is {0}".format(error)
  outfile.flush()
  outfile.close()


def spatialPredictiveCodingOption2(frames,output_file_name,width,height):
  # A Predictor
  frameCount = len(frames)
  current_working_dir = os.getcwd()
  outfile = open( output_file_name, 'w' )  
  error = 0;

  newFrames = []

  for k in range(0, frameCount):
    newFrame = []
    for i in range(0, height):
      newRow = []
      for j in range(0, width):
        if(j == 0):
          newRow.append(frames[k][i][0])
        else:
          newError = int(frames[k][i][j]) - int(frames[k][i][j-1])
          error = error + abs(newError)
          newRow.append(newError)
      newFrame.append(newRow)
    newFrames.append(newFrame)

  #for k in range(0, 3):
    #outfile.write(str(frames[k]))
    #outfile.write("\n")
    #print frames[k]
    #print newFrames[k]
    #print "\n"

  for k in range(0, frameCount):
    for i in range (0, height):
      for j in range(0, width):
        outfile.write(str(newFrames[k][i][j]))
        outfile.write(" ")
    outfile.write("\n")

  print "Total absolute prediction error is {0}".format(error)
  outfile.flush()
  outfile.close()




def spatialPredictiveCodingOption3(frames,output_file_name,width,height):
  # B Predictor
  frameCount = len(frames)
  current_working_dir = os.getcwd()
  outfile = open( output_file_name, 'w' )  
  error = 0;

  newFrames = []

  for k in range(0, frameCount):
    newFrame = []
    for i in range(0, height):
      newRow = []
      for j in range(0, width):
        if(i == 0):
          newRow.append(frames[k][0][j])
        else:
          newError = frames[k][i][j] - frames[k][i-1][j]
          error = error + abs(newError)
          newRow.append(newError)
      newFrame.append(newRow)
    newFrames.append(newFrame)

  #for k in range(0, 3):
    #outfile.write(str(frames[k]))
    #outfile.write("\n")
    #print frames[k]
    #print newFrames[k]
    #print "\n"


  for k in range(0, frameCount):
    for i in range (0, height):
      for j in range(0, width):
        outfile.write(str(newFrames[k][i][j]))
        outfile.write(" ")
    outfile.write("\n")

  print "Total absolute prediction error is {0}".format(error)
  outfile.flush()
  outfile.close()




def spatialPredictiveCodingOption4(frames,output_file_name,width,height):
  # C Predictor

  frameCount = len(frames)
  current_working_dir = os.getcwd()
  outfile = open( output_file_name, 'w' )  
  error = 0;

  newFrames = []

  for k in range(0, frameCount):
    newFrame = []
    for i in range(0, height):
      newRow = []
      for j in range(0, width):
        if i == 0 or j == 0:
          if i == 0 and j != 0:
            newRow.append(frames[k][0][j])
          elif i != 0 and j == 0:
            newRow.append(frames[k][i][0])
          elif i == 0 and j == 0:
            newRow.append(frames[k][0][0])
        else:
          newError = frames[k][i][j] - frames[k][i-1][j-1]
          error = error + abs(newError)
          newRow.append(newError)
      newFrame.append(newRow)
    newFrames.append(newFrame)


  #for k in range(0, 3):
    #outfile.write(str(frames[k]))
    #outfile.write("\n")
    #print frames[k]
    #print newFrames[k]
    #print "\n"

  for k in range(0, frameCount):
    for i in range (0, height):
      for j in range(0, width):
        outfile.write(str(newFrames[k][i][j]))
        outfile.write(" ")
    outfile.write("\n")

  print "Total absolute prediction error is {0}".format(error)
  outfile.flush()
  outfile.close()

  

def spatialPredictiveCodingOption5(frames,output_file_name,width,height):
  frameCount = len(frames)
  current_working_dir = os.getcwd()
  outfile = open( output_file_name, 'w' )  
  error = 0;
  alpha1 = 0.33
  alpha2 = 0.33
  alpha3 = 0.33


  newFrames = []

  for k in range(0, frameCount):
    newFrame = []
    for i in range (0, height):
      newRow = []
      for j in range(0, width):
        if i == 0 or j==0:
          if i == 0 and j != 0:
            newRow.append(frames[k][0][j])
          elif i != 0 and j == 0:
            newRow.append(frames[k][i][0])
          elif i == 0 and j == 0:
            newRow.append(frames[k][0][0])
        elif j == 1 or j == 2 or j == 3:
          alpha1 = 0.33
          alpha2 = 0.33
          alpha3 = 0.33

          predictorA = frames[k][i][j-1] * alpha1
          predictorB = frames[k][i-1][j] * alpha2
          predictorC = frames[k][i-1][j-1] * alpha3

          #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
          totalPredictor = predictorA + predictorB + predictorC
          newError = frames[k][i][j] - totalPredictor
          #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

          #newError = predictorA + predictorB + predictorC
          error = error + abs(newError)
          newRow.append(newError)
        else:
          predictedValOne = frames[k][i][j-3]
          predictedValOneA = frames[k][i][j-4]
          predictedValOneB = frames[k][i-1][j-3]
          predictedValOneC = frames[k][i-1][j-4]

          predictedValTwo = frames[k][i][j-2]
          predictedValTwoA = frames[k][i][j-3]
          predictedValTwoB = frames[k][i-1][j-2]
          predictedValTwoC = frames[k][i-1][j-3]

          predictedValThree = frames[k][i][j-1]
          predictedValThreeA = frames[k][i][j-2]
          predictedValThreeB = frames[k][i-1][j-1]
          predictedValThreeC = frames[k][i-1][j-2]

          #print "Hit here!!"

          #a = np.array([[3,1], [1,2]])
          #print a

          a = np.array([[predictedValOneA, predictedValOneB, predictedValOneC],[predictedValTwoA, predictedValTwoB, predictedValTwoC],[predictedValThreeA, predictedValThreeB, predictedValThreeC]])
          #print a
          #print "\n\n"
          b = np.array([predictedValOne, predictedValTwo, predictedValThree])
          #print b

          

          #+++++++++++++++++++++++++++++++++++
          if np.linalg.cond(a) < 1/sys.float_info.epsilon:
            #i = linalg.inv(x)
            x = np.linalg.solve(a, b)
            alpha1 = x[0]
            alpha2 = x[1]
            alpha3 = x[2]

            if alpha1 >= 0.0 and alpha1 < 1.0:
              if alpha2 >= 0.0 and alpha2 < 1.0:
                if alpha3 >= 0.0 and alpha3 < 1.0:
                  sumAlpha = alpha1 + alpha2 + alpha3

                  if sumAlpha != 1.0:
                    alpha1 = 0.33
                    alpha2 = 0.33
                    alpha3 = 0.33
            else:
              alpha1 = 0.33
              alpha2 = 0.33
              alpha3 = 0.33
          else:
            #handle it
            alpha1 = 0.33
            alpha2 = 0.33
            alpha3 = 0.33

          #+++++++++++++++++++++++++++++++++++

          predictorA = frames[k][i][j-1] * alpha1
          predictorB = frames[k][i-1][j] * alpha2
          predictorC = frames[k][i-1][j-1] * alpha3
          #
          totalPredictor = predictorA + predictorB + predictorC
          newError = frames[k][i][j] - totalPredictor
          #

          #newError = predictorA + predictorB + predictorC
          error = error + abs(newError)
          newRow.append(newError)

      newFrame.append(newRow)
    newFrames.append(newFrame)

  for k in range(100, 103):
    #outfile.write(str(frames[k]))
    #outfile.write("\n")
    print frames[k]
    print newFrames[k]
    #print "\n"

  for k in range(0, frameCount):
    for i in range (0, height):
      for j in range(0, width):
        outfile.write(str(newFrames[k][i][j]))
        outfile.write(" ")
    outfile.write("\n")

  print "Total absolute prediction error is {0}".format(error)
  outfile.flush()
  outfile.close()



      














main()
  