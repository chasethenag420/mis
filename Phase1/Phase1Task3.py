import cv2
import sys
import collections
import numpy as np
import colorsys as cs
import helpers as hl
import os

# Used to show input prompt component names
def get_user_color_model_components(color_model_code) :
  if color_model_code == '1':
    color_model = "RGB"
    comp1 = "R"
    comp2 = "G"
    comp3 = "B"
  elif color_model_code == '2':
    color_model = "XYZ"
    comp1 = "X"
    comp2 = "Y"
    comp3 = "Z"
  elif color_model_code == '3':
    color_model = "Lab"
    comp1 = "L"
    comp2 = "a"
    comp3 = "b"
  elif color_model_code == '4':
    color_model = "YUV"
    comp1 = "Y"
    comp2 = "U"
    comp3 = "V"
  elif color_model_code == '5':
    color_model = "YCbCr"
    comp1 = "Y"
    comp2 = "Cb"
    comp3 = "Cr"
  elif color_model_code == '6':
    color_model = "YIQ"
    comp1 = "Y"
    comp2 = "I"
    comp3 = "Q"
  elif color_model_code == '7':
    color_model = "HSL"
    comp1 = "H"
    comp2 = "S"
    comp3 = "L"
  elif color_model_code == '8':
    color_model = "HSV"
    comp1 = "H"
    comp2 = "S"
    comp3 = "V"
  else :
    print 'Not valid color model. Rerun program and choose between 1 to 8\n'
    sys.exit(1)
  return (color_model, comp1, comp2, comp3)

#Used to get input from command prompt for color instances
def get_color_comp_from_user(prefix,components):
  first = float(raw_input('Enter {0} {1} Component: '.format(prefix, components[0])))
  second = float(raw_input('Enter {0} {1} Component: '.format(prefix, components[1])))
  third = float(raw_input('Enter {0} {1} Component: '.format(prefix, components[2]))) 
  return (first,second,third)

# Splits the color instances into equal partitions from low value to mid to high values and returns it
def get_partitions(low, mid, high, total_num_of_partitions):
  low_partitions = None
  high_partitions = None
  size_on_each_size = float(total_num_of_partitions/2.0)
  low_partition_size = (mid - low)/size_on_each_size
  high_partition_size = (high - mid)/size_on_each_size
  if low_partition_size != 0 :
    low_partitions = np.arange(low, mid+low_partition_size, low_partition_size)
  else :
    low_partitions = np.ones(size_on_each_size+1) * low

  if high_partition_size != 0 :
    high_partitions = np.arange(mid, high+high_partition_size, high_partition_size)
  else :
    high_partitions = np.ones(size_on_each_size +1) * mid

  partitions = np.concatenate([low_partitions,high_partitions[1:]])[0:total_num_of_partitions+1]
  return partitions

# splits the range -1 to 1  into equal width partitions
def get_partitions_normalized(low,high,no_of_partitions):
  partition_size = ( high - low ) / float(no_of_partitions)
  if partition_size != 0 :
    partitions = np.arange(low, high+partition_size, partition_size)[0:no_of_partitions+1]
  else :
    partitions = np.ones(no_of_partitions+1) * low
  return partitions

# Creates the colormap values and associates to bins range
def genereate_colormap_and_bins(partitions,color_map_partitions,partition_size):
  color_map = {}
  color_map_bin_ranges = {}

  for idx, partition in enumerate(partitions,) :  
    if idx < len(partitions) -1 :
      current_partition = (color_map_partitions[idx]/2)
      next_partition = (color_map_partitions[idx+1]/2)
      color_map[idx+1] = tuple(np.add(current_partition, next_partition) )
      color_map_bin_ranges[idx+1] = '({0},{1})'.format(partition,partition + partition_size)
  return (color_map,color_map_bin_ranges)

# saves the colormap and bins into a file and returns the image
def save_colormap_and_bins(colormap_bins,filename,image,color_model,num_of_bits):
  
  perm_X1 = 80
  perm_X2 = 200
  y1 = 10
  y2 = 30

  file_name = '{0}'.format(filename)
  file_handle = open(file_name,'w')
  sortedBinList = sorted(colormap_bins[1].items())
  file_handle.write("{0};{1}\n".format(num_of_bits,color_model))
  for idx, elem in sorted(colormap_bins[0].items()):
    bin=sortedBinList[idx-1]
    file_handle.write("{0}:{1},{2},{3}:{4}\n".format(idx,str(elem[0]),str(elem[1]),str(elem[2]),str(bin[1])))
    cv2.rectangle(image,(perm_X1,y1),(perm_X2,y2), hl.get_color_values_in_rgb(elem[0],elem[1],elem[2],color_model), -1, 4 )
    cv2.putText(image, str(bin[1]), (perm_X2+50, y1/2+y2/2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    y1 = y2
    y2 += 20
  file_handle.close()
  print "ColorMap is saved in: "+file_name
  return image

# creates a empty image with all white pixels for given height and width
def create_blank_image(height,width):
  blank_image = np.zeros((height,width,3), np.uint8)
  blank_image[:,:] = (255,255,255)
  return blank_image

# The entry point which prompts for user input and controls the execution visualizes the colormap created and waits for user to press
# a key to closes all the windows
def main():

  cit_low = -1
  ci0_mid = 0
  ciT_high = 1
    
  small_value = 0.000000000001
  current_working_dir = os.getcwd()
  # Read the color mode from user input
  color_model_code = raw_input("""Select a Color Model:\n
  Press 1 for RGB \n
  Press 2 for XYZ \n
  Press 3 for Lab \n
  Press 4 for YUV \n
  Press 5 for YCbCr \n
  Press 6 for YIQ \n
  Press 7 for HSL\n
  Press 8 for HSV \n
  Model: """)

  color_model_data = get_user_color_model_components(color_model_code)
  color_model = color_model_data[0]
  comp1 = color_model_data[1]
  comp2 = color_model_data[2]
  comp3 = color_model_data[3]

  print "You have selected " + color_model + " Color Model\n"

  # Read the input color instance from user
  cit = get_color_comp_from_user('cit', (comp1,comp2,comp3))
  ci0 = get_color_comp_from_user('ci0', (comp1,comp2,comp3))
  ciT = get_color_comp_from_user('ciT', (comp1,comp2,comp3))
  
  # calculate total number of color instances to be created
  num_of_bits = int( raw_input( 'Enter the Number of bits <b>:' ) )
  total_num_of_color_instances = 2 ** num_of_bits

  print 'Total of 2^{0} color instance would be created\n'.format(num_of_bits)
  partition_size = ( ciT_high - cit_low ) / float(total_num_of_color_instances)
  partitions = get_partitions_normalized(cit_low,ciT_high,total_num_of_color_instances)

  # get the parititon ranges for each component
  first_partitions = get_partitions(cit[0], ci0[0], ciT[0],total_num_of_color_instances)
  second_partitions = get_partitions(cit[1], ci0[1], ciT[1],total_num_of_color_instances)
  third_partitions = get_partitions(cit[2], ci0[2], ciT[2],total_num_of_color_instances)

  #Opencv uses BGR so handle it as special case
  if color_model != "RGB" :
    color_map_partitions  = np.r_[first_partitions[None,:],second_partitions[None,:],third_partitions[None,:]]
    color_map_partitions  = color_map_partitions.transpose()
  else :
    color_map_partitions  = np.r_[third_partitions[None,:],second_partitions[None,:],first_partitions[None,:]]
    color_map_partitions  = color_map_partitions.transpose().astype(int)
  # create a blank image dimensions based on the number of color instances to be displayed
  # multiple of 21 as each instance occupy atleast 21 pixel in height
  height = total_num_of_color_instances * 21
  width = 1080
  image = create_blank_image(height,width)
  
  # save the color map to file with name created from colormodel name, color instance values and number of bits
  filename = '{0}_{1}-{2}-{3}_{4}-{5}-{6}_{7}-{8}-{9}_{10}.txt'.format(color_model,cit[0],cit[1],cit[2],ci0[0],ci0[1],ci0[2],ciT[0],ciT[1],ciT[2],num_of_bits)
  color_map_filename = '{0}\{1}'.format(current_working_dir, filename)
  colormap_bins = genereate_colormap_and_bins(partitions,color_map_partitions,partition_size)
  image = save_colormap_and_bins(colormap_bins,color_map_filename,image,color_model,num_of_bits)
  # create named window to display image
  cv2.namedWindow('ColorMap in '+color_model, cv2.WINDOW_AUTOSIZE)
  
  # display image to user
  cv2.imshow('ColorMap in '+color_model,image) 
  # save image to disc in current working directory
  cv2.imwrite('%s.jpg' % filename,image)
  # wait for user to press some key to close windows and exit
  c = cv2.waitKey(0)
  if 'q' == chr(c & 255):
    cv2.destroyAllWindows()

main()