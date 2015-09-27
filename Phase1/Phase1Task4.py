import sys
import cv2
import numpy as np
import helpers as hl

# gives the image at the given frame number for the specified video file
def get_frame(full_path,frame_number):
  image = []
  success = False
  videosource = cv2.VideoCapture(full_path)
  if videosource.isOpened() :
    for i in range( 1, frame_number + 1 ) :
      success,image = videosource.read()

    if success :
      videosource.release()
      return image
    else :
      print "Cannot read video capture object from %s. Quitting..." % videosource
      sys.exit(1)
  else :
    print "Cannot read video capture object from %s. Quitting..." % videosource
    sys.exit(1)

# convert given frame to grayscale image and save to disc
def get_frame_and_save_as_grayscale(full_path,frame_number):
  gray_image = cv2.cvtColor(get_frame(full_path,frame_number),cv2.COLOR_BGR2GRAY)
  cv2.imwrite('file1_%d_gray.jpg' % frame_number,gray_image)
  cv2.imshow('Frame {0}'.format(frame_number),gray_image)
  print 'Extracted Grayscale frame {0} saved as <file1_{0}_gray.jpg>'.format(frame_number)
  return gray_image

# Read the color map saved in task3 and give back number of bits, color model and colormap
def read_color_map(colormap_file_name):
  num_of_bits = 0
  color_map={}
  color_map_file_handle = open( colormap_file_name, 'r' )
  for line in color_map_file_handle:
    if not ":" in line :
      line=line.rstrip().split(';')
      num_of_bits = int(line[0])
      color_model = str(line[1])
    else :
      color_index=line.rstrip().split(':')
      color_map[int(color_index[0])]=color_index[1]
  return (num_of_bits,color_map,color_model)

# Creates a blank image and sets the pixels based on the grayscale differences calculated and apply given color_map and returns the image
def apply_color_map(rescaled_grayscale_diff, color_map,partitions,color_model):
  rows,cols=rescaled_grayscale_diff.shape
  height = rows
  width = cols
  blank_image = np.zeros((height,width,3), np.uint8)
  for row in range(rows):
    row_bin_indexes = np.digitize(rescaled_grayscale_diff[row],partitions)
    for col in range(cols):
      color_instance = list(int(round(float(i))) for i in color_map[row_bin_indexes[col]].split(','))
      color_instance =  hl.get_color_values_in_bgr(color_instance[0],color_instance[1],color_instance[2],color_model)
      blank_image[row][col] = np.array(color_instance).astype(np.ndarray)
  return blank_image

# gives the array with values of equal range for given bits
def get_partitions_normalized(low,high,num_of_bits):
  total_num_of_color_instances = 2 ** num_of_bits
  partition_size = ( high - low ) / float(total_num_of_color_instances)
  if partition_size != 0 :
    partitions = np.arange(low, high+partition_size, partition_size)[0:total_num_of_color_instances+1]
  else :
    partitions = np.ones(total_num_of_color_instances+1) * low
  return partitions

# The entry point which prompts for user input to get video file name, colormap file name, frame numbers.
# Controls the program execution calculates the graysacle differences and visualizes the image that got created using the color map chosen
# waits for user to press a key to clean and exit the program
def main():
  cit = -1
  cio = 0
  ciT = 1
  small_value = 0.000000000001
  max_value = 255
  video_dir = raw_input("Enter the path of the video files:\n")
  print 'All the videos will be read from %s directory' % video_dir

  video_file_name = raw_input("Enter the video file name:\n")
  frame_number_1 = int(raw_input("Enter the 1st frame number:\n"))
  frame_number_2 = int(raw_input("Enter the 2nd frame number:\n"))
  colormap_file_name = raw_input("Enter the color map file name:\n")

  print 'Extracting the grayscale for frame {0}  and {1}'.format(frame_number_1,frame_number_2)

  full_path = r'{0}\{1}'.format(video_dir,video_file_name)

  gray_image1 = get_frame_and_save_as_grayscale(full_path,frame_number_1)
  gray_image2 = get_frame_and_save_as_grayscale(full_path,frame_number_2)

  print 'Determining the Grayscale difference image'
  grayscale_diff = cv2.subtract(gray_image2.astype(np.int16),gray_image1.astype(np.int16))
  grayscale_abs_diff = cv2.subtract(gray_image2,gray_image1)
  cv2.imshow('Grayscale diff',grayscale_abs_diff)
  cv2.imwrite('Grayscale_diff_{0}_{1}.jpg'.format(frame_number_1,frame_number_2), grayscale_abs_diff)
  print 'Computation of grayscale difference image done'

  rescaled_grayscale_diff = grayscale_diff / float(max_value)  
  print 'Rescaling of grayscale difference image done'

  num_of_bits,color_map,color_model = read_color_map(colormap_file_name)
  partitions = get_partitions_normalized(cit,ciT,num_of_bits)
  recolored_image = apply_color_map(rescaled_grayscale_diff, color_map,partitions,color_model)
  print 'Recoloring of Grayscale Difference image done'

  print 'Check out the visualization'
  cv2.imshow('Recolored Grayscale Difference',recolored_image) 
  cv2.imwrite('Recolored_Grayscale_diff_{0}_{1}.jpg'.format(frame_number_1,frame_number_2), recolored_image)
  c = cv2.waitKey(0)
  if 'q' == chr(c & 255):
    cv2.destroyAllWindows()

main()