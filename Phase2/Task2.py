import cv2
import sys
import numpy as np
import os

def main():
  width=10
  height=10

  #video_dir = "sampleDataP1"
  #video_file_name = "1.mp4"

  fileSuffix=".mp4"
  video_dir = raw_input("Enter the video file directory:\n")
  video_file_name = raw_input("Enter the video file name (without the .mp4 suffix):\n")
  xMin=int(raw_input("Enter the top-left x coordiante:\n"))
  yMin=int(raw_input("Enter the top-left y coordiante:\n"))
  print "Select any one of the following: "
  print "Press 1 for No PC"
  print "Press 2 for Predictor A"
  print "Press 3 for Predictor B"
  print "Press 4 for Predictor C"
  print "Press 5 for Alpha-based Predictor"


  option_number=raw_input("Enter the option number:\n")

  full_video_name = video_file_name + ".mp4"
  full_path = r'{0}/{1}'.format(video_dir,full_video_name) 
  output_video_file_name='extracted_'+full_video_name
  output_video_full_path=r'{0}/{1}'.format(video_dir,output_video_file_name)
  output_video_full_path='output.mp4'
  
  output_file_name=r'{0}_{1}.spc'.format(video_file_name,option_number)

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

def spatialPredictiveCodingOption1(frames,output_file_name,width,height):
  # No Predictive Coding
  frameCount = len(frames)
  current_working_dir = os.getcwd()
  outfile = open( output_file_name, 'w' )  
  error = 0
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
  error = 0
  extraError=0

  newFrames = []
  for k in range(0, frameCount):
    newFrame = []
    for i in range(0, height):
      newRow = []
      for j in range(0, width):
        if(j == 0):
          newRow.append(frames[k][i][0])
          extraError=extraError+frames[k][i][0]
        else:
          newError = int(frames[k][i][j]) - int(frames[k][i][j-1])
          error = error + newError
          newRow.append(newError)
      newFrame.append(newRow)
    newFrames.append(newFrame)
  for k in range(0, frameCount):
    for i in range (0, height):
      for j in range(0, width):
        outfile.write(str(newFrames[k][i][j]))
        outfile.write(" ")
    outfile.write("\n")

  print "Total absolute prediction error is {0}".format(abs(error-extraError))
  outfile.flush()
  outfile.close()




def spatialPredictiveCodingOption3(frames,output_file_name,width,height):
  # B Predictor
  frameCount = len(frames)
  current_working_dir = os.getcwd()
  outfile = open( output_file_name, 'w' )  
  error = 0
  extraError=0

  newFrames = []

  for k in range(0, frameCount):
    newFrame = []
    for i in range(0, height):
      newRow = []
      for j in range(0, width):
        if(i == 0):
          newRow.append(frames[k][0][j])
          extraError=extraError+frames[k][0][j]
        else:
          newError = int(frames[k][i][j]) - int(frames[k][i-1][j])
          error = error + newError
          newRow.append(newError)
      newFrame.append(newRow)
    newFrames.append(newFrame)

  for k in range(0, frameCount):
    for i in range (0, height):
      for j in range(0, width):
        outfile.write(str(newFrames[k][i][j]))
        outfile.write(" ")
    outfile.write("\n")

  print "Total absolute prediction error is {0}".format(abs(error-extraError))
  outfile.flush()
  outfile.close()




def spatialPredictiveCodingOption4(frames,output_file_name,width,height):
  # C Predictor

  frameCount = len(frames)
  current_working_dir = os.getcwd()
  outfile = open( output_file_name, 'w' )  
  error = 0
  extraError=0

  newFrames = []

  for k in range(0, frameCount):
    newFrame = []
    for i in range(0, height):
      newRow = []
      for j in range(0, width):
        if i == 0 or j == 0:
          if i == 0 and j != 0:
            newRow.append(frames[k][0][j])
            extraError=extraError+frames[k][0][j]
          elif i != 0 and j == 0:
            newRow.append(frames[k][i][0])
            extraError=extraError+frames[k][i][0]
          elif i == 0 and j == 0:
            newRow.append(frames[k][0][0])
            extraError=extraError+frames[k][0][0]
        else:
          newError = frames[k][i][j] - frames[k][i-1][j-1]
          error = error + newError
          newRow.append(newError)
      newFrame.append(newRow)
    newFrames.append(newFrame)

  for k in range(0, frameCount):
    for i in range (0, height):
      for j in range(0, width):
        outfile.write(str(newFrames[k][i][j]))
        outfile.write(" ")
    outfile.write("\n")

  print "Total absolute prediction error is {0}".format(abs(error-extraError))
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

          totalPredictor = predictorA + predictorB + predictorC
          newError = float(frames[k][i][j]) - float(totalPredictor)

          error = error + newError
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

          a = np.array([[predictedValOneA, predictedValOneB, predictedValOneC],[predictedValTwoA, predictedValTwoB, predictedValTwoC],[predictedValThreeA, predictedValThreeB, predictedValThreeC]])
          b = np.array([predictedValOne, predictedValTwo, predictedValThree])

          if np.linalg.cond(a) < 1/sys.float_info.epsilon:
            x = np.linalg.solve(a, b)
            alpha1 = x[0]
            alpha2 = x[1]
            alpha3 = x[2]

            if alpha1 >= 0.0 and alpha1 <= 1.0:
              if alpha2 >= 0.0 and alpha2 <= 1.0:
                if alpha3 >= 0.0 and alpha3 <= 1.0:
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

          predictorA = frames[k][i][j-1] * alpha1
          predictorB = frames[k][i-1][j] * alpha2
          predictorC = frames[k][i-1][j-1] * alpha3

          totalPredictor = predictorA + predictorB + predictorC
          newError = float(frames[k][i][j]) - float(totalPredictor)

          error = error + newError
          newRow.append(newError)

      newFrame.append(newRow)
    newFrames.append(newFrame)

  for k in range(0, frameCount):
    for i in range (0, height):
      for j in range(0, width):
        outfile.write(str(newFrames[k][i][j]))
        outfile.write(" ")
    outfile.write("\n")

  print "Total absolute prediction error is {0}".format(abs(error-extraError))
  outfile.flush()
  outfile.close()


main()
  