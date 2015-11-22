__author__ = 'Darius'
import cv2
from sys import platform as _platform
import math

def extract_video_portion(fullPath,width,height,numOfBits,outFileName):
  frames = None
  frameWidth=None
  frameHeight=None
  outfile = open( outFileName,'w' )
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

  while cap.isOpened:
    success, img = cap.read()
    if success == True :
      yuvImage = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
      y,u,v=cv2.split(yuvImage)
      blocksCoordinates={}
      for i in range(0,frameWidth,width):
        for j in range(0,frameHeight,height):
          yChannel=y[j:j+height,i:i+width]
          count+=1
          blocksCoordinate = (i, j)
          dct(yChannel, numOfBits, outfile, frameId, blocksCoordinate)


      frameId=frameId+1
      outfile.flush()
    else:
      break

  outfile.close()

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

if __name__ == "__main__":
  width = 8;
  height = 8;
  fileSuffix = ".mp4"
  if _platform == "linux" or _platform == "linux2":
    slash = '/'
  elif _platform == "darwin":
    slash = '/'
  elif _platform == "win32":
    slash = '\\'

  videoDir = raw_input("Enter the video file directory:\n")
  videoFileName = raw_input("Enter the video file name:\n")
  numOfBits=int(raw_input("Enter number of bits:\n"))

  #numOfBits = 2
  #videoDir=r'F:\ASU_Projects\MIS\mis\Phase3\reducedSizeVideo'
  #videoFileName='R1'

  fullPath = '{0}{2}{1}'.format(videoDir, videoFileName+fileSuffix, slash)
  outFileName = '{0}_blockdct_{1}.bct'.format(videoFileName, numOfBits)
  extract_video_portion(fullPath,width,height,numOfBits,outFileName)
