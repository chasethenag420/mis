import cv2
import collections
from sys import platform as _platform
__author__ = 'Monken'


# Transform an height by width pixel block using Discrete Wavelet Transform
def block_dwt_transform(y_channel, height, width, num_components, frame_id, out_file):
  input_frame = y_channel.tolist()                # translate the Y channel numpy array to a python list
  row_transform = []                              # for storing for results of row transform
  k = 0                                           # counter for column transform operation
  for i in range(height):
    row = input_frame[i]
    output_row = []                               # empty the 1st half of the output row to build the next transform row
    output_row2 = []                              # empty the 2nd half of the output row to build the next transform row
    for j in range(0, width-1, 2):                # traverse the row in steps of 2
      avg_val = (row[j] + row[j+1]) / 2           # determine the average of the current and the next values
      diff_val = (row[j] - row[j+1]) / 2          # determine the difference of the current and next values
      if not output_row:                          # if the first half of the output row is empty
        output_row = [avg_val]                    # set it to the average value
      else:                                       # otherwise
        output_row.append(avg_val)                # add the average value to the first half of the output row
      if not output_row2:                         # if the second half of the output row is empty
        output_row2 = [diff_val]                  # set it to the diff value
      else:                                       # otherwise
       output_row2.append(diff_val)               # add the diff value to the second half of the output row
    output_row = output_row + output_row2         # concatenate the first and second halves of the output row
    if not row_transform:                         # if the row transform block is empty
      row_transform = [output_row]                # set it to the output row
    else:                                         # otherwise
      row_transform.append(output_row)            # add the output row to it
  transform_block = [[0 for x in range(width)] for x in range(height)]    # create a height by width output array
  for i in range(0, height-1, 2):                 # traverse the row transformed block in steps of 2
    row1 = row_transform[i]                       # get the 1st input column value
    row2 = row_transform[i + 1]                   # get the 2nd input column value
    row1a = transform_block[k]                    # get the 1st half of output column
    row5 = transform_block[k + (height/2)]        # get the 2nd half of output column
    for j in range(width):                        # traverse the group by column
      row1a[j] = (row1[j] + row2[j]) / 2          # set the 1st half of column to the average of the values
      row5[j] = (row1[j] - row2[j]) / 2           # set the 2nd half of column to the difference of the values
    k += 1
  flattened_block = [item for sublist in transform_block for item in sublist]   # flatten the transform block
  frequency_list = collections.Counter(flattened_block)                         # build freq list from flattened list
  n_freq_list = list(reversed(frequency_list.most_common()[:num_components]))   # get m most sig wavelet components
  for i in range(len(n_freq_list)):
    row = n_freq_list[i]
    component_id = row[0]
    component_freq = row[1]
    out_file.write("{0},{1},{2}\n".format(frame_id, component_id, component_freq))


def extract_video(full_path, num_components, out_file_name):
  frames = None
  frame_id = 0
  out_file = open(out_file_name, 'w')
  cap = cv2.VideoCapture(full_path)
  if cap.isOpened == None:
    return frames
  if cv2.__version__ == '3.0.0':
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  else:
    frame_width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
  while cap.isOpened:
    success, img = cap.read()
    if success:
      yuv_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
      y_channel, u, v = cv2.split(yuv_image)
      block_dwt_transform(y_channel, frame_height, frame_width, num_components, frame_id, out_file)
      frame_id += 1
      out_file.flush()
    else:
      break
  out_file.close()

if __name__ == "__main__":
  slash = '\''
  if _platform == "linux" or _platform == "linux2":
    slash = '/'
  elif _platform == "darwin":
    slash = '/'
  elif _platform == "win32":
    slash = '\\'
  file_suffix = ".mp4"
  video_dir = raw_input("Enter the video file directory:\n")
  video_file_name = raw_input("Enter the video file name:\n")
  num_components = int(raw_input("Enter number of significant wavelet components:\n"))
  #video_dir = r'F:\\GitHub\\mis\\Phase3\\reducedSizeVideo'     # for testing
  #video_file_name = 'R1'                                      # for testing
  #num_components = 4                                          # for testing
  full_path = '{0}{2}{1}'.format(video_dir, video_file_name + file_suffix, slash)
  out_file_name = '{0}_framedwt_{1}.fwt'.format(video_file_name, num_components)
  extract_video(full_path, num_components, out_file_name)