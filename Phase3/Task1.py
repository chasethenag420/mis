import cv2
import math
import numpy as np
import collections
import os.path

from sys import platform as _platform


def quantize(yChannel, numOfBits, frameId, blocksCoordinate, outfile,maxValue,minValue):
    numOfPartitions = 2 ** numOfBits
    signal= yChannel.tolist()[0]
    partitionSize = (maxValue - minValue)/float(numOfPartitions)
    partitions=None
    if partitionSize != 0:
      partitions = np.arange(minValue, maxValue+partitionSize, partitionSize)
    else :
      partitions = np.ones(numOfPartitions+1) * minValue
    binIndexes=np.digitize(np.array(signal),partitions)
    representative=[]
    for value in range(1,len(partitions)):
      representative.append(int(partitions[value-1]/2+partitions[value]/2))
    quantized=[]
    for i in binIndexes.tolist():
      index=0
      if i-1 >= len(representative):
        index= i-2
      else:
        index=i-1
      quantized.append(int(representative[index]))
    frequency_list=collections.Counter(quantized)
    if len(frequency_list.keys()) != len(representative):
      for i in representative:
        if i not in frequency_list.keys():
          frequency_list[i]=0
    for idx, value in enumerate(frequency_list.keys()):
      freq_val = frequency_list[value]
      outfile.write("{0},{1},{2},{3}\n".format(frameId, blocksCoordinate, value, freq_val))


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
def block_dwt_transform(y_channel, size, num_comp, frame_id, block_coordinates, out_file):
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
    out_file.write("{0},{1},{2},{3}\n".format(frame_id, block_coordinates, comp_id, comp_val))


def dct(yChannel, numOfBits, outfile, frameId, blocksCoordinate):
  u_v_Image = []
  for u in range(0, 8):
    newRow = []
    for v in range(0, 8):
      Cu = 0.0
      Cv = 0.0
      sumVar = 0.0
      if u == 0:
        Cu = math.sqrt(2)/2
      else:
        Cu = 1.0
      if v == 0:
        Cv = math.sqrt(2)/2
      else:
        Cv = 1.0
      for x in range(0, 8):
        for y in range(0, 8):
          pix = float(yChannel[x][y])
          cosX = math.cos(((2*x + 1) * u * math.pi)/16.0)
          cosY = math.cos(((2*y + 1) * v * math.pi)/16.0)
          curr = pix * cosX * cosY
          sumVar += curr
      newVal = 0.25 * Cu * Cv * sumVar
      newRow.append(newVal)
    u_v_Image.append(newRow)
  zigZag(u_v_Image, numOfBits, outfile, frameId, blocksCoordinate)
  return


'''
This code was adapted from http://ideone.com/xKRTwU
'''
def zigZag(u_v_Image, numOfBits, outfile, frameId, blocksCoordinate):
  rows = 8
  cols = 8
  count = 0
  direction = 1
  r = 0
  c = 0
  while r < rows and c < cols:
    if count == numOfBits:
      return
    freq_comp_id = r*8 + c
    freq_val = u_v_Image[r][c]
    outfile.write('{0},{1},{2},{3},{4}\n'.format(frameId, blocksCoordinate[0], blocksCoordinate[1], freq_comp_id, freq_val))
    if direction == 1:
      if c == cols - 1:
        r += 1
        direction = -1
      elif r == 0:
        c += 1
        direction = -1
      else:
        r -= 1
        c += 1
    else:
      if r == rows - 1:
        c += 1
        direction = 1
      elif c == 0:
        r += 1
        direction = 1
      else:
        c -= 1
        r += 1
    count += 1

def extract_video_portion(fullPath,width,height,numOfBits,a_outFileName,b_outFileName,c_outFileName,d_outFileName):
  frames = None
  frameWidth=None
  frameHeight=None
  a_outfile=None
  b_outfile=None
  c_outfile=None
  d_outfile=None

  if not os.path.isfile(a_outFileName):
    a_outfile = open( a_outFileName,'w' )
  if not os.path.isfile(b_outFileName):
    b_outfile = open( b_outFileName,'w' )
  if not os.path.isfile(c_outFileName):
    c_outfile = open( c_outFileName,'w' )
  if not os.path.isfile(d_outFileName):
    d_outfile = open( d_outFileName,'w' )

  if a_outfile==None and b_outfile==None and c_outfile==None and d_outfile==None:
    return

  cap = cv2.VideoCapture(fullPath)
  if cap.isOpened == None:
    return frames
  if cv2.__version__=='3.0.0':
    frameWidth=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  else:
    frameWidth=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    frameHeight=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
  frameId=0
  frames={}
  prevFrame=None
  while cap.isOpened:
    success, img = cap.read()
    if success == True :
      yuvImage = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
      y,u,v=cv2.split(yuvImage)
      blocksCoordinates={}
      for i in range(0,frameWidth,width):
        for j in range(0,frameHeight,height):
          yChannel=y[j:j+height,i:i+width]
          blocksCoordinate='{0},{1}'.format(i, j)
          blockCoordinates = (i, j)

          flatYChannel = np.reshape(yChannel, (1, width*height))
          quantize(flatYChannel,numOfBits,frameId,blocksCoordinate,a_outfile,255,0)                      # Task 1a
          dct(yChannel, numOfBits, b_outfile, frameId, blockCoordinates)                             # Task 1b
          block_dwt_transform(yChannel, height, numOfBits, frameId, blocksCoordinate, c_outfile)     # Task 1c
          if frameId !=0:
            prevFrameYChannel=prevFrame[j:j+height,i:i+width]
            diffYChannel=cv2.subtract(yChannel.astype(np.int16),prevFrameYChannel.astype(np.int16))
            flatDiffChannel = np.reshape(diffYChannel, (1, width*height))
            quantize(flatDiffChannel, numOfBits, frameId-1, blocksCoordinate, d_outfile,255,-255)             # Task 1d
          else:
            prevFrame=y

      frameId=frameId+1
      a_outfile.flush()
      b_outfile.flush()
      c_outfile.flush()
      d_outfile.flush()
    else:
      break
  a_outfile.close()
  b_outfile.close()
  c_outfile.close()
  d_outfile.close()
  cap.release()


if __name__ == "__main__":
  width=8
  height=8
  if _platform == "linux" or _platform == "linux2":
    slash = '/'
  elif _platform == "darwin":
    slash = '/'
  elif _platform == "win32":
    slash = '\\'
  fileSuffix=".mp4"
  videoDir = raw_input("Enter the video file directory:\n")
  videoFileName = raw_input("Enter the video file name:\n")
  numOfBits=int(raw_input("Enter number of bits:\n"))
  #numOfBits=4
  #videoDir=r'F:\\GitHub\\mis\\Phase3\\reducedSizeVideo'
  #videoFileName='R1'
  fullPath = '{0}{2}{1}'.format(videoDir, videoFileName+fileSuffix, slash)
  a_outFileName='{0}_hist_{1}.hst'.format(videoFileName,numOfBits)
  b_outFileName = '{0}_blockdct_{1}.bct'.format(videoFileName, numOfBits)
  c_outFileName = '{0}_blockdwt_{1}.bwt'.format(videoFileName, numOfBits)
  d_outFileName='{0}_diff_{1}.dhc'.format(videoFileName,numOfBits)
  extract_video_portion(fullPath, width, height, numOfBits, a_outFileName,b_outFileName,c_outFileName,d_outFileName)
