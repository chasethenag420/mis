import cv2
import math
import numpy as np
import collections
from sys import platform as _platform


def d_quantize(yChannel, numOfBits, frameId, blocksCoordinate, outfile):
    numOfPartitions = 2 ** numOfBits
    signal= yChannel.tolist()[0]
    #signal=[1,2,3,4,5,6,7,8,9]
    #maxValue = max(signal)
    #minValue = min(signal)
    maxValue=255
    minValue=-255
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


# Transform an height by width pixel block using Discrete Wavelet Transform
def block_dwt_transform(y_channel, height, width, num_components, frame_id, block_coordinates, out_file):
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
  transform_block = [[0 for x in range(width)] for x in range(height)]   # create a hght x wdth output array
  for i in range(0, height-1, 2):           # traverse the row transformed block in steps of 2
    row1 = row_transform[i]                       # get the 1st input column value
    row2 = row_transform[i + 1]                   # get the 2nd input column value
    row1a = transform_block[k]                    # get the 1st half of output column
    row5 = transform_block[k + (height/2)]  # get the 2nd half of output column
    for j in range(width):                  # traverse the group by column
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
    out_file.write("{0},{1},{2},{3}\n".format(frame_id, block_coordinates, component_id, component_freq))


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


def a_quantize(yChannel, numOfBits, frameId, blocksCoordinate, outfile):
    numOfPartitions = 2 ** numOfBits
    signal= yChannel.tolist()[0]
    #signal=[1,2,3,4,5,6,7,8,9]
    #maxValue = max(signal)
    #minValue = min(signal)
    maxValue=255
    minValue=0
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


def extract_video_portion(fullPath,width,height,numOfBits,a_outFileName,b_outFileName,c_outFileName,d_outFileName):
  frames = None
  frameWidth=None
  frameHeight=None
  a_outfile = open( a_outFileName,'w' )
  b_outfile = open( b_outFileName,'w' )
  c_outfile = open( c_outFileName,'w' )
  d_outfile = open( d_outFileName,'w' )
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
  count=0
  frames={}
  prevFrame=None
  while cap.isOpened:
    success, img = cap.read()
    if success == True :
      yuvImage = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
      y,u,v=cv2.split(yuvImage)
      blocksCoordinates={}
      if frameId==0:
        prevFrame=y
        frameId=frameId+1
        continue
      else:
        for i in range(0,frameWidth,width):
          for j in range(0,frameHeight,height):
            yChannel=y[j:j+height,i:i+width]
            prevFrameYChannel=prevFrame[j:j+height,i:i+width]
            count+=1
            #diffYChannel=cv2.absdiff(yChannel,prevFrameYChannel)
            diffYChannel=cv2.subtract(yChannel.astype(np.int16),prevFrameYChannel.astype(np.int16))
            blocksCoordinate='{0},{1}'.format(i, j)
            blockCoordinates = (i, j)
            flatYChannel = np.reshape(diffYChannel, (1, width*height))
            flatDiffChannel = np.reshape(diffYChannel, (1, width*height))
            a_quantize(flatYChannel,numOfBits,frameId,blocksCoordinate,a_outfile)                              # Task 1a
            dct(yChannel, numOfBits, b_outfile, frameId, blockCoordinates)                                     # Task 1b
            block_dwt_transform(yChannel, height, width, numOfBits, frameId, blocksCoordinate, c_outfile)      # Task 1c
            d_quantize(flatDiffChannel, numOfBits, frameId-1, blocksCoordinate, d_outfile)                     # Task 1d
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
  #videoDir = raw_input("Enter the video file directory:\n")
  #videoFileName = raw_input("Enter the video file name:\n")
  #numOfBits=int(raw_input("Enter number of bits:\n"))
  numOfBits=4
  videoDir=r'F:\\GitHub\\mis\\Phase3\\reducedSizeVideo'
  videoFileName='R1'
  fullPath = '{0}{2}{1}'.format(videoDir, videoFileName+fileSuffix, slash)
  a_outFileName='{0}_hist_{1}.hst'.format(videoFileName,numOfBits)
  b_outFileName = '{0}_blockdct_{1}.bct'.format(videoFileName, numOfBits)
  c_outFileName = '{0}_blockdwt_{1}.bwt'.format(videoFileName, numOfBits)
  d_outFileName='{0}_diff_{1}.dhc'.format(videoFileName,numOfBits)
  extract_video_portion(fullPath, width, height, numOfBits, a_outFileName,b_outFileName,c_outFileName,d_outFileName)
