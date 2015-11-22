import cv2
import Task1a
import Task1b
import Task1d
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
    distance += getEucledianDistance(frame1[i],frame2[i])
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
  totalFrames = len(lines)/frameRowSize
  similarity={}

  for currentFrameId in range(0,totalFrames):
    if currentFrameId != frameId:
      currentFrame=getFrameById(lines,currentFrameId,frameRowSize)
      similarity[currentFrameId]=getSimilarity(currentFrame,queryFrame)

  sorted_similarity = sorted(similarity.items(), key=operator.itemgetter(1),reverse=True)

  return sorted_similarity

def visualizeFrames(fullPath,sorted_similarity,queryFrameId, prefix):
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
          count1 +=1
      elif count2 < 1 and frameId==queryFrameId:
          cv2.imshow("{1} Query Frame:{0}".format(frameId,prefix),img)
          count2 +=1
      else:
        break

    frameId += 1

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
  frameId=raw_input("Enter the frame id:\n");
  numOfBits=int(raw_input("Enter number of bits n:\n"))
  numOfSignComp=int(raw_input("Enter number of Significant Components m:\n"))

  #numOfBits=1
  #numOfSignComp=2
  #videoDir=r'F:\ASU_Projects\MIS\mis\Phase3\reducedSizeVideo'
  #videoFileName='R2'
  #frameId=0

  fullPath = '{0}{2}{1}'.format(videoDir, videoFileName+fileSuffix, slash)
  tasks=['1a', '1b', '1c', '1d', '2']

  for index, task_name in enumerate(tasks):
    outFileName = getOutFileName(videoFileName, numOfBits, numOfSignComp, task_name)
    if task_name == '1a':
      Task1a.extract_video_portion(fullPath, width, height, numOfBits, outFileName)
      sorted_similarity=getSimilarFrames(outFileName,frameId)
      visualizeFrames(fullPath, sorted_similarity, frameId,"Task1a")
    elif task_name == '1b':
      Task1b.extract_video_portion(fullPath, width, height, numOfBits, outFileName)
      sorted_similarity=getSimilarFrames(outFileName, frameId)
      visualizeFrames(fullPath, sorted_similarity, frameId, "Task1b")
    elif task_name == '1d':
      Task1d.extract_video_portion(fullPath, width, height, numOfBits, outFileName)
      sorted_similarity=getSimilarFrames(outFileName, frameId)
      visualizeFrames(fullPath, sorted_similarity, frameId, "Task1d")


  c = cv2.waitKey(0)
  if 'q' == chr(c & 255):
    cv2.destroyAllWindows()