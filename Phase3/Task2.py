import cv2
import math
from sys import platform as _platform
__author__ = 'Monken'


def haar_transform(input_block):
  haar_coeff = 0.7071
  size = len(input_block)
  top_half = []
  bottom_half = []
  for i in range(0, size-1, 2):
    output_row1_left = []
    output_row1_right = []
    output_row2_left = []
    output_row2_right = []
    row1 = input_block[i]
    row2 = input_block[i+1]
    for j in range(0, size-1, 2):
      top_left = haar_coeff**2 * (row1[j] + row1[j+1] + row2[j] + row2[j+1])
      top_right = haar_coeff**2 * (row1[j] - row1[j+1] + row2[j] - row2[j+1])
      bottom_left = haar_coeff**2 * (row1[j] + row1[j+1] - row2[j] - row2[j+1])
      bottom_right = haar_coeff**2 * (row1[j] - row1[j+1] - row2[j] + row2[j+1])
      if not output_row1_left:                             # build left half of top
        output_row1_left = [top_left]
      else:
        output_row1_left.append(top_left)
      if not output_row1_right:                            # build right half of top
        output_row1_right = [top_right]
      else:
        output_row1_right.append(top_right)
      if not output_row2_left:                             # build left half of bottom
        output_row2_left = [bottom_left]
      else:
        output_row2_left.append(bottom_left)
      if not output_row2_right:                            # build right half of bottom
        output_row2_right = [bottom_right]
      else:
        output_row2_right.append(bottom_right)
    output_row1 = output_row1_left + output_row1_right
    if not top_half:                                       # build top half
      top_half = [output_row1]
    else:
      top_half.append(output_row1)
    output_row2 = output_row2_left + output_row2_right
    if not bottom_half:                                    # build bottom half
      bottom_half = [output_row2]
    else:
      bottom_half.append(output_row2)
  block = top_half + bottom_half                           # build transform block
  return block


# Transform an height by width pixel block using Discrete Wavelet Transform
def block_dwt_transform(y_channel, size, num_comp, frame_id, out_file):
  y_channel = y_channel.astype(float)
  input_frame = y_channel.tolist()                # translate the Y channel numpy array to a python list
  transform_block = haar_transform(input_frame)   # first haar wavelet transform
  k = size / 2                                    # counter for rest of haar wavelet transforms
  while k > 1:
    input_block = []
    for i in range(k):                            # get the next block to be transformed
      row = transform_block[i]
      input_row = []
      for j in range(k):
        if not input_row:
          input_row = [row[j]]
        else:
          input_row.append(row[j])
      if not input_block:
        input_block = [input_row]
      else:
        input_block.append(input_row)
    next_block = haar_transform(input_block)
    for i in range(k):                            # put the transformed smaller block back into the larger block
      row = next_block[i]
      row1 = transform_block[i]
      for j in range(k):
        row1[j] = row[j]
    k /= 2                                        # get next smaller haar wavelet transform block size
  for i in range(len(transform_block)):
    row = transform_block[i]
    for j in range(len(row)):
      row[j] = int(round(row[j], 0))
  target_size = int(math.ceil(num_comp**0.5))     # change transform block back to rounded integers
  x = 1
  sig_comp_list = []
  for i in range(target_size):
    row = transform_block[i]
    for j in range(target_size):
      if not sig_comp_list:
        sig_comp_list = ([[x, row[j]]])
      else:
        sig_comp_list.append([x, row[j]])
      x += 1
  for i in range(len(sig_comp_list)):
    row = sig_comp_list[i]
    comp_id = row[0]
    comp_val = row[1]
    out_file.write("{0},{1},{2}\n".format(frame_id, comp_id, comp_val))


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
      block_dwt_transform(y_channel, frame_height, num_components, frame_id, out_file)
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