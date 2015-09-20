import cv2
import sys
import collections
import numpy as np
import colorsys as cs
import helpers as hl
from matplotlib import pyplot as plt
import operator
'''
cit_red_component = 0
cit_green_component = 0
cit_blue_component = 0

cio_red_component = 50
cio_green_component = 50
cio_blue_component = 50

ciT_red_component = 255
ciT_green_component = 0
ciT_blue_component = 0

num_of_bits = 4

color_model_code = '1'
color_model = "RGB"
comp1 = "R"
comp2 = "G"
comp3 = "B"
'''
cit = -1
cio = 0
ciT = 1
height = 720
width = 480
blank_image = np.zeros((height,width,3), np.uint8)
blank_image[:,:] = (255,255,255)
perm_X1 = 80
perm_X2 = 200
y1 = 45
y2 = 70
small_value = 0.000000000001

def get_color_values( first, second, third, color_model):
  if color_model == "RGB":
    return (first,second,third)
  elif color_model == "XYZ":
    return hl.xyz2rgb(first,second,third)
  elif color_model == "Lab":
    return hl.lab2rgb(first,second,third)
  elif color_model == "YUV":
    return hl.yuv2rgb(first,second,third)
  elif color_model == "YCbCr":
    return hl.ycbcr2rgb(first,second,third)
  elif color_model == "YIQ":
    return hl.yiq2rgb(first,second,third)
  elif color_model == "HSL":
    return hl.hsl2rgb(first,second,third)
  elif color_model == "HSV":
    return hl.hsv2rgb(first,second,third)
  else :
    return (first,second,third)

color_model = ""
comp1 = ""
comp2 = ""
comp3 = ""
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
  print 'Not valid color model. Rerun program and choose between 1 to 7\n'
  sys.exit(1)


print "You have selected " + color_model + " Color Model\n"
cit_red_component   = float(raw_input('Enter Component Cit ' + comp1 + ' Component: '))  #beginning of if block
cit_green_component = float(raw_input('Enter Component Cit ' + comp2 + ' Component: '))
cit_blue_component  = float(raw_input('Enter Component Cit ' + comp3 + ' Component: '))

print 'cit( {0},{1},{2} )'.format( cit_red_component, cit_green_component, cit_blue_component )
cio_red_component   = float(raw_input('Enter Component Cio ' + comp1 + ' Component: '))
cio_green_component = float(raw_input('Enter Component Cio ' + comp2 + ' Component: '))
cio_blue_component  = float(raw_input('Enter Component Cio ' + comp3 + ' Component: '))

print 'cio( {0},{1},{2} )'.format( cio_red_component, cio_green_component, cio_blue_component )
ciT_red_component   = float(raw_input('Enter Component CiT ' + comp1 + ' Component: '))
ciT_green_component = float(raw_input('Enter Component CiT ' + comp2 + ' Component: '))
ciT_blue_component  = float(raw_input('Enter Component CiT ' + comp3 + ' Component:'))

print 'ciT( {0},{1},{2} )'.format( ciT_red_component, ciT_green_component, ciT_blue_component ) #end of if block

num_of_bits = int( raw_input( 'Enter the Number of bits <b>:' ) )

total_num_of_color_instances        = 2 ** num_of_bits

num_of_color_instances_in_each_side = total_num_of_color_instances / 2
print 'Total of 2^{0} color instance would be created\n'.format(num_of_bits)

partition_size = ( ciT - cit ) / float(total_num_of_color_instances)
partitions = np.arange(cit, ciT+small_value, partition_size)


red_low_partition_size    = (cio_red_component - cit_red_component) / float(num_of_color_instances_in_each_side)
green_low_partition_size  = (cio_green_component - cit_green_component) / float(num_of_color_instances_in_each_side)
blue_low_partition_size   = (cio_blue_component - cit_blue_component) / float(num_of_color_instances_in_each_side)
red_high_partition_size   = (ciT_red_component - cio_red_component) / float(num_of_color_instances_in_each_side)
green_high_partition_size = (ciT_green_component - cio_green_component) / float(num_of_color_instances_in_each_side)
blue_high_partition_size  = (ciT_blue_component - cio_blue_component) / float(num_of_color_instances_in_each_side)


if red_low_partition_size != 0 :
  red_low_patitions = np.arange(cit_red_component, cio_red_component+red_low_partition_size, red_low_partition_size)
else :
  red_low_patitions = np.ones(num_of_color_instances_in_each_side +1) * cio_red_component

if green_low_partition_size != 0 :
  green_low_patitions = np.arange(cit_green_component, cio_green_component+green_low_partition_size, green_low_partition_size)
else :
  green_low_patitions = np.ones(num_of_color_instances_in_each_side+1) * cio_green_component

if blue_low_partition_size != 0 :
  blue_low_patitions = np.arange(cit_blue_component, cio_blue_component+blue_low_partition_size, blue_low_partition_size)
else :
  blue_low_patitions = np.ones(num_of_color_instances_in_each_side+1) * cio_blue_component

if red_high_partition_size != 0 :
  red_high_patitions = np.arange(cio_red_component, ciT_red_component+red_high_partition_size, red_high_partition_size)
else :
  red_high_patitions = np.ones(num_of_color_instances_in_each_side+1) * ciT_red_component

if green_high_partition_size != 0 :
  green_high_patitions = np.arange(cio_green_component, ciT_green_component+green_high_partition_size, green_high_partition_size)
else :
  green_high_patitions = np.ones(num_of_color_instances_in_each_side+1) * ciT_green_component

if blue_high_partition_size != 0 :
  blue_high_patitions = np.arange(cio_blue_component, ciT_blue_component+blue_high_partition_size, blue_high_partition_size)
else :
  blue_high_patitions = np.ones(num_of_color_instances_in_each_side+1) * ciT_blue_component

red_partitions        = np.concatenate([red_low_patitions,red_high_patitions[1:]])[0:total_num_of_color_instances+1]
green_partitions      = np.concatenate([green_low_patitions,green_high_patitions[1:]])[0:total_num_of_color_instances+1]
blue_partitions       = np.concatenate([blue_low_patitions,blue_high_patitions[1:]])[0:total_num_of_color_instances+1]
# print red_partitions
# print green_partitions
# print blue_partitions

# print red_low_patitions
# print red_high_patitions
# print green_low_patitions
# print green_high_patitions
# print blue_low_patitions
# print blue_high_patitions

#Opencv uses BGR so handle it as special case
if color_model != "RGB" :
  color_map_partitions  = np.r_[red_partitions[None,:],green_partitions[None,:],blue_partitions[None,:]]
  color_map_partitions  = color_map_partitions.transpose()
else :
  color_map_partitions  = np.r_[blue_partitions[None,:],green_partitions[None,:],red_partitions[None,:]]
  color_map_partitions  = color_map_partitions.transpose().astype(int)



color_map = {}
color_map_bin_ranges = {}

for idx, partition in enumerate(partitions) :  
  if idx < len(partitions) -1 :
    current_partition = (color_map_partitions[idx]/2)
    next_partition = (color_map_partitions[idx+1]/2)
    color_map[idx+1] = tuple(np.add(current_partition, next_partition) )
    color_map_bin_ranges[idx+1] = '({0},{1})'.format(partition,partition + partition_size)
sortedColorList = sorted(color_map.items())
sortedBinList = sorted(color_map_bin_ranges.items())

file_name = '{0}_{1}-{2}-{3}_{4}-{5}-{6}_{7}-{8}-{9}_{10}.txt'.format(color_model,cit_red_component,cit_green_component,cit_blue_component,cio_red_component,cio_green_component,cio_blue_component,ciT_red_component,ciT_green_component,ciT_blue_component,num_of_bits)
file_handle = open(file_name,'w')

bins_file_name = '{0}_{1}-{2}-{3}_{4}-{5}-{6}_{7}-{8}-{9}_{10}.txt_bins'.format(color_model,cit_red_component,cit_green_component,cit_blue_component,cio_red_component,cio_green_component,cio_blue_component,ciT_red_component,ciT_green_component,ciT_blue_component,num_of_bits)
bins_file_handle = open(bins_file_name,'w')

for idx, elem in sortedColorList:
  cv2.rectangle(blank_image,(perm_X1,y1),(perm_X2,y2), get_color_values(elem[0],elem[1],elem[2],color_model), -1, 4 )
  cv2.putText(blank_image, str(sortedBinList[idx-1][1]), (perm_X2+50, (y1+y2)/2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
  y1 = y2
  y2 += 20
  
bins_file_handle.write("{0}\n".format(num_of_bits))
for item in sortedBinList:
  bins_file_handle.write("{0} {1}\n".format(item[0],str(item[1])))
bins_file_handle.close()

for color in sortedColorList:
  file_handle.write("{0}:{1},{2},{3}\n".format(color[0],str(color[1][0]),str(color[1][1]),str(color[1][2])))
file_handle.close()

cv2.imshow(' Frame',blank_image) 

c = cv2.waitKey(0)
if 'q' == chr(c & 255):
  cv2.destroyAllWindows()
