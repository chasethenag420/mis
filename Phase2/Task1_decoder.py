import cv2
import sys
import numpy as np
import os

def main():
  video_file_name = '1'
  option_number='5'
  width=10
  height=10
  video_dir=r'F:\ASU_Projects\MIS\mis\Phase1\sampleDataP1'
  # video_file_name = raw_input("Enter the video file name:\n")
  # option_number=raw_input("Enter the option number:\n")
  input_file_name=r'{0}_{1}.tpc'.format(video_file_name,option_number)
  output_video_file_name='extracted_'+video_file_name
  output_video_full_path=r'{0}\{1}'.format(video_dir,output_video_file_name)
  output_video_full_path='output.avi'
  
  output_file_name=r'{0}_{1}.mp4'.format(video_file_name,option_number)
  current_working_dir = os.getcwd()
  infile = open( input_file_name ) 
  frames= None
  

  if input_file_name != None:
    if option_number=='1':
      frames=tpcDecodingOption1(infile) 
    elif option_number=='2':
      frames=tpcDecodingOption2(infile)
    elif option_number=='3':
      frames=tpcDecodingOption3(infile)
    elif option_number=='4':
      frames=tpcDecodingOption4(infile)
    elif option_number=='5':
      frames=tpcDecodingOption5(infile)
    else: 
      print "Input not valid"
      quit()
  else:
    print "Some error while reading video file"

  if frames !=None:
    decodeVideo(frames,width,height,output_video_full_path,input_file_name)

  infile.flush()
  infile.close()

# frames: each row represent pixels in a frame which will be reshaped to width and height
def decodeVideo(frames,width,height,output_video_full_path,input_file_name):
  frameSize=frames.shape
  frameRate=30
  output_file_name=input_file_name+"decoded.txt"
  outfile = open( output_file_name, 'w' )
  #fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
  #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  fourcc = cv2.VideoWriter_fourcc('I', 'Y', 'U', 'V')
  out = cv2.VideoWriter(output_video_full_path, fourcc, frameRate,(width,height))
  for x in range(0,frameSize[0]):    
      # for y in range(0,frameSize[1]):
      # pixel=[lines[x][y],0,0]
      # row=[]      
      # for i in range(0,width):
      #   col=[]
      #   for j in range(0,height):          
      #     col.append(pixel)
      #   row.append(col)
      # frame=np.array(row)
      frame=np.array(np.array(frames[x][:]).reshape(width,height), dtype=np.uint8)
      cv2.imshow("Decoded Y channel",frame)
      outfile.write(" ".join(map(str,frames[x][:]))+"\n")
      #out.write(frame)
      c = cv2.waitKey(1)
      if 'q' == chr(c & 255):
        break     
  out.release()    
  # cv2.imshow("frame",frames)
  #print frames.shape
  # wait for user to press some key to close windows and exit
  #c = cv2.waitKey(0)
  #if 'q' == chr(c & 255):
  cv2.destroyAllWindows()
  print "Output saved to text "+output_file_name
  outfile.flush()
  outfile.close()

  
def tpcDecodingOption1(infile):
   
  lines=None
  for line in infile:
    if lines==None:
      lines=np.array(list(map(int,line.split())))
    else:
      lines=np.column_stack((lines,list(map(int,line.split()))))
  return lines
  

def tpcDecodingOption2(infile):
  lines=None
  for line in infile:
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


def tpcDecodingOption3(infile):
  lines=None
  for line in infile:
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

def tpcDecodingOption4(infile):
  lines=None
  alpha1=0.5
  alpha2=0.5
  for line in infile:
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
  
def tpcDecodingOption5(infile):
  lines=None
  alpha1=float(raw_input("Enter aplha1 predictor: "))
  alpha2=float(raw_input("Enter aplha2 predictor: "))
  for line in infile:
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
