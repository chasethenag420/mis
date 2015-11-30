import cv2
import Task1
import Task2
import numpy as np
from sys import platform as _platform
import numpy.linalg as la
import operator
import collections
__author__ = 'Nagarjuna'


def getOutFileName(videoFileName, numOfBits, numOfSignComp, task_num):
  outFileName=None

  if task_num=='1a':
    outFileName='{0}_hist_{1}.hst'.format(videoFileName, numOfBits)
  elif task_num=='1b':
    outFileName='{0}_blockdct_{1}.bct'.format(videoFileName, numOfBits)
  elif task_num=='1c':
    outFileName='{0}_blockdwt_{1}.bwt'.format(videoFileName, numOfBits)
  elif task_num=='1d':
    outFileName='{0}_diff_{1}.dhc'.format(videoFileName, numOfBits)
  elif task_num=='2':
    outFileName='{0}_framedwt_{1}.fwt'.format(videoFileName, numOfSignComp)

  return outFileName


def getFrameData(outFileName,frameId):
  outFile = open(outFileName,'r')
  inputFrame=[]
  lines=[]
  for line in outFile:
    line = [int(float(i)) for i in line.split(",")]
    lines.append(line)
    if line[0] == frameId:
      inputFrame.append(line)

  outFile.close()
  return lines,inputFrame

def getEucledianDistance(vector1,vector2):
  return la.norm(np.array(vector1)-np.array(vector2));

def getSimilarity(frame1,frame2):
  frameRowSize=len(frame1)
  distance=0
  for i in range(0,frameRowSize):
    distance += getEucledianDistance(frame1[i][1:],frame2[i][1:])
  avgDistance = float(distance)/frameRowSize
  similarity = 1/(1+ avgDistance)
  return similarity


def getFrameById(lines,frameId,frameRowSize):
  frameStartIndex = frameId*frameRowSize
  frameEndIndex = frameStartIndex + frameRowSize
  frame = lines[frameStartIndex:frameEndIndex]
  return frame

def getSimilarFrames(outFileName,frameId):
  lines, queryFrame= getFrameData(outFileName,frameId)
  frameRowSize= len(queryFrame)
  if frameRowSize == 0:
    return None
  totalFrames = len(lines)/frameRowSize
  similarity={}

  for currentFrameId in range(0,totalFrames):
    if currentFrameId != frameId:
      currentFrame=getFrameById(lines,currentFrameId,frameRowSize)
      similarity[currentFrameId]=getSimilarity(currentFrame,queryFrame)

  sorted_similarity = sorted(similarity.items(), key=operator.itemgetter(1),reverse=True)

  return sorted_similarity



def quantize(yChannel, numOfBits, frameId, blocksCoordinate,maxValue,minValue,queryDiffFrames):
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
      queryDiffFrames.append([frameId, blocksCoordinate[0],blocksCoordinate[1], value, freq_val])



def getSimilarFramesForDiffQuantization(outFileName,fullPath,frameId,width,height,numOfBits):
  lines, queryFrame= getFrameData(outFileName,frameId)

  frameRowSize= len(queryFrame)
  if frameRowSize == 0:
    return None
  totalFrames = len(lines)/frameRowSize

  similarity={}
  queryFrame=getVideoFrameById(fullPath,frameId)
  for currentFrameId in range(0,totalFrames-1):
    if currentFrameId != frameId:
      currentFrame=getFrameById(lines,currentFrameId,frameRowSize)
      currentFrame2=getFrameById(lines,currentFrameId+1,frameRowSize)
      queryDiffFrames=getQuantizedDiff(fullPath,currentFrameId,queryFrame,width,height,numOfBits)
      similarity[currentFrameId]=getSimilarity(currentFrame+currentFrame2,queryDiffFrames)

  sorted_similarity = sorted(similarity.items(), key=operator.itemgetter(1),reverse=True)

  return sorted_similarity

def getQuantizedDiff(fullPath,frameId,queryFrame,width,height,numOfBits):

  queryDiffFrames=[]
  #queryFrame=getVideoFrameById(fullPath,QueryFrameId)
  prevFrame=getVideoFrameById(fullPath,frameId)
  nextFrame=getVideoFrameById(fullPath,frameId+2)
  if queryFrame== None or prevFrame==None or nextFrame==None:
    return None

  y=queryFrame
  frameWidth=y.shape[0]
  frameHeight=y.shape[1]
  for i in range(0,frameWidth,width):
    for j in range(0,frameHeight,height):
      yChannel=y[j:j+height,i:i+width]
      blockCoordinates = (i, j)

      prevFrameYChannel=prevFrame[j:j+height,i:i+width]
      diffYChannel=cv2.subtract(yChannel.astype(np.int16),prevFrameYChannel.astype(np.int16))
      flatDiffChannel = np.reshape(diffYChannel, (1, width*height))
      quantize(flatDiffChannel, numOfBits, frameId, blockCoordinates,255,-255,queryDiffFrames)

  y=nextFrame
  prevFrame=queryFrame
  for i in range(0,frameWidth,width):
    for j in range(0,frameHeight,height):
      yChannel=y[j:j+height,i:i+width]
      blockCoordinates = (i, j)

      prevFrameYChannel=prevFrame[j:j+height,i:i+width]
      diffYChannel=cv2.subtract(yChannel.astype(np.int16),prevFrameYChannel.astype(np.int16))
      flatDiffChannel = np.reshape(diffYChannel, (1, width*height))
      quantize(flatDiffChannel, numOfBits, frameId+1, blockCoordinates,255,-255,queryDiffFrames)

  return queryDiffFrames

def getVideoFrameById(fullPath,inputFrameId):
  frame=None
  cap = cv2.VideoCapture(fullPath)
  if cap.isOpened == None:
    return None
  frameId=0
  while cap.isOpened:
    success, img = cap.read()
    if success == True :
      if frameId != inputFrameId:
        frameId +=1
        continue
      else:
        yuvImage = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y,u,v=cv2.split(yuvImage)
        frame=y
        break
  cap.release()
  return frame


def visualizeFrames(fullPath,sorted_similarity,queryFrameId, prefix):
  if sorted_similarity==None:
    return None
  topTen = dict(sorted_similarity[:10])
  for value in sorted_similarity[:10]:
    print "{2} Matching score with Frame {0}: {1}".format(value[0], value[1], prefix)

  cap = cv2.VideoCapture(fullPath)

  if cap.isOpened == None:
    return None
  frameId=0

  count1=0
  count2=0
  while cap.isOpened:
    success, img = cap.read()
    if success == True :
      if count1 < 10 and frameId != queryFrameId:
        if (frameId in topTen.keys()):
          cv2.imshow("{1} Frame:{0}".format(frameId,prefix),img)
          cv2.imwrite("{1}_Frame_{0}.jpg".format(frameId,prefix),img)
          count1 +=1
      elif count2 < 1 and frameId==queryFrameId:
          cv2.imshow("{1} Query Frame:{0}".format(frameId,prefix),img)
          cv2.imwrite("{1}_Query_Frame_{0}.jpg".format(frameId,prefix),img)
          count2 +=1
    else:
       break

    frameId += 1

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
  frameId=int(raw_input("Enter the frame id:\n"))
  numOfBits=int(raw_input("Enter number of bits n:\n"))
  numOfSignComp=int(raw_input("Enter number of Significant Components m:\n"))

  #numOfBits=1
  #numOfSignComp=2
  #videoDir=r'F:\ASU_Projects\MIS\mis\Phase3\reducedSizeVideo'
  #videoFileName='R2'
  #frameId=0

  fullPath = '{0}{2}{1}'.format(videoDir, videoFileName+fileSuffix, slash)
  tasks=['1a', '1b', '1c', '1d', '2']

  a_outFileName='{0}_hist_{1}.hst'.format(videoFileName,numOfBits)
  b_outFileName = '{0}_blockdct_{1}.bct'.format(videoFileName, numOfBits)
  c_outFileName = '{0}_blockdwt_{1}.bwt'.format(videoFileName, numOfBits)
  d_outFileName='{0}_diff_{1}.dhc'.format(videoFileName,numOfBits)
  Task1.extract_video_portion(fullPath, width, height, numOfBits, a_outFileName,b_outFileName,c_outFileName,d_outFileName)

  for index, task_name in enumerate(tasks):
    if task_name == '1a':
      sorted_similarity=getSimilarFrames(a_outFileName,frameId)
      visualizeFrames(fullPath, sorted_similarity, frameId,"Task1a")
    elif task_name == '1b':
      sorted_similarity=getSimilarFrames(b_outFileName, frameId)
      visualizeFrames(fullPath, sorted_similarity, frameId, "Task1b")
    elif task_name == '1c':
      sorted_similarity=getSimilarFrames(c_outFileName, frameId)
      visualizeFrames(fullPath, sorted_similarity, frameId, "Task1c")
    elif task_name == '1d':
      sorted_similarity=getSimilarFramesForDiffQuantization(d_outFileName,fullPath,frameId,width,height,numOfBits)
      visualizeFrames(fullPath, sorted_similarity, frameId, "Task1d")
    elif task_name == '2':
      outFileName = getOutFileName(videoFileName, numOfBits, numOfSignComp, task_name)
      Task2.extract_video(fullPath, numOfSignComp, outFileName)
      sorted_similarity=getSimilarFrames(outFileName, frameId)
      visualizeFrames(fullPath, sorted_similarity, frameId, "Task2")


  c = cv2.waitKey(0)
  if 'q' == chr(c & 255):
    cv2.destroyAllWindows()