import cv2
import sys
import numpy as np
import os

def spatialPredictiveDecodingOption1(frames, output_file_name):
	#no PC
	current_working_dir = os.getcwd()
  	outfile = open( output_file_name, 'w' )
  	frameCount = len(frames)

	for k in range(0, frameCount):
		for i in range(0, 10):
			for j in range(0, 10):
				outfile.write(str(frames[k][i][j]))
				outfile.write(" ")
		outfile.write("\n")


	newFrames = np.array(frames)
	
	outfile.flush()
  	outfile.close()
  	print "\nOutput saved to {0}\n".format(output_file_name)
	return newFrames

def spatialPredictiveDecodingOption2(frames, output_file_name):
	frameCount = len(frames)
	current_working_dir = os.getcwd()
  	outfile = open( output_file_name, 'w' )

	for k in range(0,frameCount):
		newFrame = []
		for i in range(0, 10):
			newRow = []
			for j in range(0, 10):
				if(j == 0):
					frames[k][i][j] = int(round(frames[k][i][j]))
				else:
					frames[k][i][j] = frames[k][i][j] + frames[k][i][j-1]
					frames[k][i][j] = int(round(frames[k][i][j]))

	for k in range(0, frameCount):
		for i in range (0, 10):
			for j in range(0, 10):
				outfile.write(str(frames[k][i][j]))
				outfile.write(" ")
		outfile.write("\n")


	newFrames = np.array(frames)

	outfile.flush()
  	outfile.close()
  	print "\nOutput saved to {0}\n".format(output_file_name)
	return newFrames


def spatialPredictiveDecodingOption3(frames, output_file_name):
	frameCount = len(frames)
	current_working_dir = os.getcwd()
  	outfile = open( output_file_name, 'w' )
	
	for k in range(0,frameCount):
		newFrame = []
		for i in range(0, 10):
			newRow = []
			for j in range(0, 10):
				if(i == 0):
					frames[k][i][j] = int(round(frames[k][i][j]))
				else:
					frames[k][i][j] = frames[k][i][j] + frames[k][i-1][j]
					frames[k][i][j] = int(round(frames[k][i][j]))


	for k in range(0, frameCount):
		for i in range (0, 10):
			for j in range(0, 10):
				outfile.write(str(frames[k][i][j]))
				outfile.write(" ")
		outfile.write("\n")

	newFrames = np.array(frames)
	
	outfile.flush()
  	outfile.close()
  	print "\nOutput saved to {0}\n".format(output_file_name)
	return newFrames

	

def spatialPredictiveDecodingOption4(frames, output_file_name):
	frameCount = len(frames)
	current_working_dir = os.getcwd()
  	outfile = open( output_file_name, 'w' )
	
	for k in range(0,frameCount):
		newFrame = []
		for i in range(0, 10):
			newRow = []
			for j in range(0, 10):
				if i == 0 or j == 0:
					if i == 0 and j != 0:
						frames[k][i][j] = int(round(frames[k][0][j]))
					elif i != 0 and j == 0:
						frames[k][i][j] = int(round(frames[k][i][0]))
					elif i == 0 and j == 0:
						frames[k][i][j] = int(round(frames[k][0][0]))
				else:
					frames[k][i][j] = frames[k][i][j] + frames[k][i-1][j-1]
					frames[k][i][j] = int(round(frames[k][i][j]))

	for k in range(0, frameCount):
		for i in range (0, 10):
			for j in range(0, 10):
				outfile.write(str(frames[k][i][j]))
				outfile.write(" ")
		outfile.write("\n")


	newFrames = np.array(frames)

	outfile.flush()
  	outfile.close()
  	print "\nOutput saved to {0}\n".format(output_file_name)
	return newFrames

def spatialPredictiveDecodingOption5(frames, output_file_name):
	frameCount = len(frames)
	current_working_dir = os.getcwd()
  	outfile = open( output_file_name, 'w' )

	for k in range(0, frameCount):
		for i in range(0, 10):
			for j in range(0, 10):
				if i == 0 or j == 0:
					if i == 0 and j != 0:
						frames[k][i][j] = int(round(frames[k][i][j]))
					elif i != 0 and j == 0:
						frames[k][i][j] = int(round(frames[k][i][j]))
					elif i == 0 and j == 0:
						frames[k][i][j] = int(round(frames[k][i][j]))
				elif j == 1 or j == 2 or j == 3:
					alpha1 = 0.33
					alpha2 = 0.33
					alpha3 = 0.33

					predictorA = frames[k][i][j-1] * alpha1
					predictorB = frames[k][i-1][j] * alpha2
					predictorC = frames[k][i-1][j-1] * alpha3
					totalPredictor = predictorA + predictorB + predictorC

					frames[k][i][j] = frames[k][i][j] + totalPredictor
					frames[k][i][j] = int(round(frames[k][i][j]))

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

					predictorA = frames[k][i][j-1] * alpha1
					predictorB = frames[k][i-1][j] * alpha2
					predictorC = frames[k][i-1][j-1] * alpha3

					totalPredictor = predictorA + predictorB + predictorC

					frames[k][i][j] = frames[k][i][j] + totalPredictor
					frames[k][i][j] = int(round(frames[k][i][j]))


	for k in range(0, frameCount):
		for i in range (0, 10):
			for j in range(0, 10):
				outfile.write(str(frames[k][i][j]))
				outfile.write(" ")
		outfile.write("\n")

	newFrames = np.array(frames)
	
	outfile.flush()
  	outfile.close()
  	print "\nOutput saved to {0}\n".format(output_file_name)
	return newFrames

    
def decodeVideo(frames,fullPath,width,height,outputVideoFileName):

	size = len(frames)
  frameRate=30
  fourcc=-1

  cap = cv2.VideoCapture(fullPath)
  outVideoFile=None
  if cap.isOpened:
    if cv2.__version__=='3.0.0':
      frameRate=cap.get(cv2.CAP_PROP_FPS)
      fourcc=cap.get(cv2.CAP_PROP_FOURCC)
    else:
      frameRate=cap.get(cv2.cv.CV_CAP_PROP_FPS)
      fourcc=cap.get(cv2.cv.CV_CAP_PROP_FOURCC)

  #fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
  #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  #fourcc = cv2.VideoWriter_fourcc('I', 'Y', 'U', 'V')
  
  outVideoFile = cv2.VideoWriter(outputVideoFileName, int(fourcc), frameRate,(width,height))
  for x in range(0,size): 
      frame=np.array(frames[x], dtype=np.uint8)
      u=np.ones((width,height), dtype=np.uint8)*128
      v=np.ones((width,height), dtype=np.uint8)*128
      yuvImage=cv2.merge((frame,u,v))
      rgbImage = cv2.cvtColor(yuvImage, cv2.COLOR_YUV2BGR)
      cv2.imshow("Decoded Y channel",yuvImage)
      outVideoFile.write(rgbImage)
      c = cv2.waitKey(1)
      if 'q' == chr(c & 255):
        break     
  outVideoFile.release()    
  cv2.destroyAllWindows()



def main():

	
	#txt_file_name = raw_input("Enter the video file name:\n")
	#txt_file_name = "1.mp4_5.spc"
	width=10
  height=10
  fileSuffix=".mp4"
  videoDir = raw_input("Enter the video file directory:\n")
  videoFileName=raw_input("Enter the video file name:\n")  
  fullPath = r'{0}/{1}'.format(videoDir,videoFileName+fileSuffix)
	outputFileName=r'{0}_{1}_out{2}'.format(videoFileName,optionNumber,fileSuffix)

	txt_file_name = raw_input("Enter the .spc file you would like to decode:\n")

	print "Select any one of the following: "
  	print "Press 1 for No PC Decoder"
  	print "Press 2 for Predictor A Decoder"
  	print "Press 3 for Predictor B Decoder"
  	print "Press 4 for Predictor C Decoder"
  	print "Press 5 for Alpha-based Predictor Decoder"

  	option_number=raw_input("Enter the option number:\n")

  	output_file_name=r'{0}_{1}_decoded.txt'.format(txt_file_name,option_number)
	
	count = 0
	
	Frames = []

	with open(txt_file_name) as openfileobject:
		for line in openfileobject:
			frame = []
			for char in line.split():
				if txt_file_name == "1_5.spc":
					number = float(char)
				else:
					number = int(char)
				frame.append(number)
				count = count + 1
				#print number
			Frames.append(frame)

	numFrames = count/100

	newFrames = []

	for k in range(0, numFrames):
		frame = Frames[k]
		newframe = []
		for i in range(0, 10):
			row = []
			multiplier = i*10
			for j in range(0, 10):
				row.append(frame[multiplier+j])
			newframe.append(row)
		newFrames.append(newframe)

	finalFrames = []

	if option_number == '1':
		finalFrames = spatialPredictiveDecodingOption1(newFrames, output_file_name)
	elif option_number == '2':
		finalFrames = spatialPredictiveDecodingOption2(newFrames, output_file_name)
	elif option_number == '3':
		finalFrames = spatialPredictiveDecodingOption3(newFrames, output_file_name)
	elif option_number == '4':
		finalFrames = spatialPredictiveDecodingOption4(newFrames, output_file_name)
	elif option_number == '5':
		finalFrames = spatialPredictiveDecodingOption5(newFrames, output_file_name)
	else:
		print "Input not valid"
		quit()

	output_video_full_path='output.avi'

	if finalFrames != None:
		

		decodeVideo(frames,fullPath,width,height,outputFileName)



main()