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
      frames=quantizationOption1(infile) 
    elif option_number=='2':
      frames=quantizationOption2(infile)
    else: 
      print "Input not valid"
      quit()
  else:
    print "Some error while reading video file"

  infile.flush()
  infile.close()

  
def quantizationOption1(infile):
   
  lines=None
  for line in infile:
    if lines==None:
      lines=np.array(list(map(int,line.split())))
    else:
      lines=np.column_stack((lines,list(map(int,line.split()))))
  return lines

main()
