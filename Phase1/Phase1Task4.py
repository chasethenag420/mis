import sys
import cv2
import numpy as np

#np.set_printoptions(threshold=np.nan)
#video_dir = raw_input("Enter the path of the video files:\n")
def getFileContentsAsString( path ) :

  content = None
  try :
    ratings_file = open( path, 'r' )
    content = ratings_file.read()
    ratings_file.close()    
  except BaseException, e:
    print 'Exception caught while reading file: %s' % e

  return content
cit = -1
cio = 0
ciT = 1
small_value = 0.000000000001
max_value = 255
video_dir = r"F:\ASU_Projects\MIS\sampleDataP1"
print 'All the videos will be read from %s directory' % video_dir

# video_file_name = raw_input("Enter the video file name:\n")
# video_frame_rate = int(raw_input("Enter the video frame rate:\n"))
# frame_number_1 = int(raw_input("Enter the 1st frame number:\n"))
# frame_number_2 = int(raw_input("Enter the 2nd frame number:\n"))
# colormap_file_name = raw_input("Enter the color map file name:\n")
video_file_name = '1.mp4'
video_frame_rate = 30
frame_number_1 = 1
frame_number_2 = 2
colormap_file_name = r"F:\ASU_Projects\MIS\RGB_0-0-0_50-50-50_255-0-0_4.txt"

print 'Extracting the grayscale for frame {0}  and {1}'.format(frame_number_1,frame_number_2)

full_path = r'{0}\{1}'.format(video_dir,video_file_name)
print 'full path %s' % full_path
videosource = cv2.VideoCapture(full_path)
image = []
for i in range( 1, frame_number_1 + 1 ) :
  success,image = videosource.read(i)

cv2.imwrite("frame%d.jpg" % 1, image)

for i in range( 1, abs(frame_number_1 - frame_number_2) + 1 ) :
  success,image = videosource.read(i)

cv2.imwrite("frame%d.jpg" % 2, image)
print type(image[0][0])
gray_image1 = cv2.imread("frame%d.jpg" % 1, cv2.IMREAD_GRAYSCALE)
gray_image2 = cv2.imread("frame%d.jpg" % 2, cv2.IMREAD_GRAYSCALE)
print type(gray_image1[0][0])
cv2.imwrite('file1_%d_gray.jpg' % frame_number_1,gray_image1)
cv2.imwrite('file1_%d_gray.jpg' % frame_number_2,gray_image2)

print 'Extracted Grayscale saves as <file1_{0}_gray.jpg> and <file1_{1}_gray.jpg>'.format(frame_number_1,frame_number_2)

cv2.imshow('Frame 1',cv2.imread('file1_%d_gray.jpg' % frame_number_1))
cv2.imshow('Frame 2',cv2.imread('file1_%d_gray.jpg' % frame_number_2))

print 'Determining the Grayscale difference image'
grayscale_diff = cv2.subtract(gray_image2.astype(np.int16),gray_image1.astype(np.int16))
cv2.imshow('Gray Scale Difference', grayscale_diff)
print 'Computation of grayscale difference image done'

rescaled_grayscale_diff = grayscale_diff / float(max_value)
print 'Rescaling of grayscale difference image done'

rows,cols=rescaled_grayscale_diff.shape

ratings_file = open( colormap_file_name+'_bins', 'r' )
num_of_bits = int(ratings_file.readline())
ratings_file.close()
color_map={}
color_map_file_handle = open( colormap_file_name, 'r' )
for line in color_map_file_handle:
  color_index=line.rstrip().split(':')
  print color_index
  color_map[int(color_index[0])]=color_index[1]
total_num_of_color_instances = 2 ** num_of_bits
partition_size = ( ciT - cit ) / float(total_num_of_color_instances)
partitions = np.arange(cit, ciT+small_value, partition_size)

height = rows
width = cols
blank_image = np.zeros((height,width,3), np.uint8)

for row in range(rows):
  row_bin_indexes = np.digitize(rescaled_grayscale_diff[row],partitions)
  for col in range(cols):
    color_instance = list( int(i) for i in color_map[row_bin_indexes[col]].split(',')) 
    blank_image[row][col] = np.array(color_instance).astype(np.ndarray)
cv2.imshow(' Frame',blank_image) 

print 'Recoloring of Grayscale Difference image done'

print 'Check out the visualization'
c = cv2.waitKey(0)
if 'q' == chr(c & 255):
  cv2.destroyAllWindows()
