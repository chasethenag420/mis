import sys
import cv2
import numpy as np
from sys import platform as _platform
import os
import math
import struct
import ast

hex2bin = dict('{:x} {:04b}'.format(x,x).split() for x in range(16))
bin2hex = dict('{:b} {:x}'.format(x,x).split() for x in range(16))

def float_dec2bin(d):
  neg = False
  if d < 0:
    d = -d
    neg = True
  hx = float(d).hex()
  p = hx.index('p')
  bn = ''.join(hex2bin.get(char, char) for char in hx[2:p])
  return (('-' if neg else '') + bn.strip('0') + hx[p:p+2] + bin(int(hx[p+2:]))[2:])

def float_bin2dec(bn):
  neg = False
  if bn[0] == '-':
    bn = bn[1:]
    neg = True
  dp = bn.index('.')
  extra0 = '0' * (4 - (dp % 4))
  bn2 = extra0 + bn
  dp = bn2.index('.')
  p = bn2.index('p')
  hx = ''.join(bin2hex.get(bn2[i:min(i+4, p)].lstrip('0'), bn2[i]) for i in range(0, dp+1, 4))
  bn3 = bn2[dp+1:p]
  extra0 = '0' * (4 - (len(bn3) % 4))
  bn4 = bn3 + extra0
  hx += ''.join(bin2hex.get(bn4[i:i+4].lstrip('0')) for i in range(0, len(bn4), 4))
  hx = (('-' if neg else '') + '0x' + hx + bn2[p:p+2] + str(int('0b' + bn2[p+2:], 2)))
  return float.fromhex(hx)


#####NEED TO FINISH#####
# get the compression model from the file name
def get_compression_model(file_name):
  input_file_name_split=file_name.split('_')
  input_file_name_split=input_file_name_split[3].split('.')
  compression_model_code=input_file_name_split[0]
  return compression_model_code

def get_file_size(path):
  fileHandle = open(path, 'rb')
  byteArr = bytearray(fileHandle.read(os.path.getsize(path)))
  fileHandle.close()
  fileSize = len(byteArr)
  return fileSize

#####NEED TO FINISH#####
# gets the data contained in the compression file  - need to finish once have info about task 3 output
def get_file(full_path, compression_model_code, input_image, input_key):
  #no compression input image - input_image = np.array([], str)
  # Shannon_fano input image - input_image = np.array([], str)
  # LZW/Dictionary input image - input_image = []
  # Arithmetic input image - input_image = []
  inFile = open( full_path,'r' )
  input_key=[]
  inputList=[]
  temp=[]
  count = 0
  width=0
  if compression_model_code=='1':
    for line in inFile:
      input_image.append(list(map(str,line.split())))

  elif compression_model_code=='2':
    for line in inFile:
      if count==0:
        temp=line.split(',')

      else:
        input_image.append(line.split())
      count=count+1
    for i in temp:
      input_key.append(tuple(map(int,i.split())))

    input_key=input_key[:len(input_key)-1]
  elif compression_model_code=='3':
    for line in inFile:
      if count==0:
        temp=line.split(',')

      else:
        input_image.append(line.split())
        #inputList.append(list(map(float,line.split())))
      count=count+1
    for i in temp:
      input_key.append(tuple(map(int,i.split())))

  elif compression_model_code=='4':
    for line in inFile:
      if count==0:
        temp=line.split(',')
      elif count==1:
        width=int(line)
      else:
        input_image.append(line.split())
        #inputList.append(list(map(float,line.split())))
      count=count+1
    for i in temp:
      x=i.split()
      if len(x)==2:
        input_key.append((int(x[0]),float(x[1])))


  return input_image, input_key,width


# convert the image from binary without any compression
def no_compression(input_image, output_image):
  #temp=[]
  #for i,value in enumerate(input_image):               # traverse the image array and
  #  temp = [int(x, 2) for x in value]                 # convert each binary value in the image to an integer
  #  if not output_image:
  #    output_image=[temp]
  #  else:
  #    output_image.append(temp)               # and add to the output image
  #output_image=np.asarray(output_image,dtype=np.uint8)
  return input_image


# create the output image using the symbol_dictionary
def create_output_image(input_image, output_image, symbol_dictionary):
  output_image = np.copy(input_image)               # copy the input image to the output image - it is an array of strings at this point
  for i in range(len(symbol_dictionary)):             # go through the dictionary
    symbol = symbol_dictionary[i]               # for each entry in the dictionary
    for h in np.nditer(output_image, op_flags=['readwrite']): # traverse the output image as read/writeable
      if h == symbol[2]:                    # for each compression symbol in the image that matches the dictionary entry
        h[...] = str(symbol[0])               # change the compression symbol to its represented value
  return output_image                       # and return the decompressed image

# get the symbol counts from the input key
def create_symbol_dictionary(input_key, symbol_dictionary):
  for i in range(len(input_key)):                           # traverse the input key
    single_key = input_key[i]                           # get each key entry
    if not symbol_dictionary:                           # if the symbol frequency list is empty
      symbol_dictionary = [(single_key[0], single_key[1], '')]      # set the key entry as its initial value and a blank place holder for their compression symbol
    else:                                     # otherwise
      symbol_dictionary.append((single_key[0], single_key[1], ''))    # add the key entry to the dictionary with a blank place holder for their compression symbol
  return symbol_dictionary

# recursive Shannon-Fano algorithm to create value symbols
def shannon_fano_algorithm(symbol_dictionary, top_index, bottom_index):
  s_mid = 0                                 # initialize current range midpoint
  size = bottom_index - top_index + 1                     # set current range size
  if size > 1:                                # while there are entries in the dictionary
    s_mid = int(size / 2 + top_index)                    # determine the mid point of the range
    for i in range(top_index, bottom_index + 1):              # for loop through
      symbol = symbol_dictionary[i]                   # symbol dictionary to build tree
      if i < s_mid:                           # for the left branch of tree
        symbol_dictionary[i] = (symbol[0], symbol[1], symbol[2] + '0')  # add next digit to the left branch's symbol
      else:                               # and for the right branch of tree
        symbol_dictionary[i] = (symbol[0], symbol[1], symbol[2] + '1')  # add next digit to the right branch's symbol
    shannon_fano_algorithm(symbol_dictionary, top_index, s_mid-1)     # recursive call for the left branch
    shannon_fano_algorithm(symbol_dictionary, s_mid, bottom_index)      # recursive call for the right branch
  return symbol_dictionary


# Shanon-Fano decompression algorithm
def shannon_fano_decompression(input_image, input_key, symbol_dictionary, output_image):
    symbol_dictionary = create_symbol_dictionary(input_key, symbol_dictionary)          # create the symbol_dictionary
    top_index = 0                                       # initial first index
    bottom_index = len(symbol_dictionary) - 1                         # initial last index
    symbol_dictionary = shannon_fano_algorithm(symbol_dictionary, top_index, bottom_index)    # call recursive shannon-fano algorithm to build compression symbols
    output_image = create_output_image(input_image, output_image, symbol_dictionary)      # create the output image
    return  output_image


# convert the LZW/Dictionary image from binary to integers
def convert_from_binary(input_image):
    integer_image = []                # initialize the empty integer image
    for i in range(len(input_image)):       # traverse the image array
        row = input_image[i]            # get each row
        integer_row = []              # reset the integer_row to empty
        for j in range(len(row)):         # traverse the row
            value = int(row[j], 2)          # convert each value in the row to an integer
            if not integer_row:           # if the integer row is empty
                integer_row = [value]       # set the value as its initial entry
            else:                 # otherwise
                integer_row.append(valeu)     # append it to the integer row
        if not integer_image:           # if the integer image is empty
            integer_image = [integer_row]     # set the integer row as it initial value
        else:                   # otherwise
            integer_image.append(integer_row)   # append the integer row to the integer image
    input_image = list(integer_image)       # copy the integer image to the output image
    return input_image

# Create the string table from the input key
def create_string_table(input_key, string_table):
    for i in range(len(input_key)):                       # traverse the input key
        single_key = input_key[i]                       # get eack key entry
        if not string_table:                          # if the symbol frequency list is empty
            string_table = [(int(single_key[0]), str(single_key[1]))]     # set the key entry as its initial value and a blank place holder for their compression symbol
        else:                                 # otherwise
            string_table.append((int(single_key[0]), str(single_key[1])))   # add the key entry to the dictionary with a blank place holder for their compression symbol
    return string_table

#####NEED TO FINISH#####
# Dictionary/LZW decompression algorithm
def dictionary_lzw_decompression(input_image, input_key, string_table, output_image):
  string_table = create_string_table(input_key, string_table)   # create the string table for the input key
  input_image = convert_from_binary(input_image)          # convert the input image from binary to integers
  new_code = len(string_table) + 1                # set the next new code
  for i in range(len(input_image)):               # traverse the input image
    s = ''                            # reset s to empty
    output_row = ''                       # reset output row to empty
    row = input_image[i]                    # get each row from the image
    for j in range(len(row)):                 # traverse the row from the next value
      k = row[j]                        # get each input code
      for l in range(len(string_table)):            # traverse the string table
        string_row = string_table[l]            # get each entry in the table
        if string_row[0] == k:                # if k matches the entry in the string table
          entry = str(string_row[1])            # get the entry value from the string table
          if not output_row:                # if row output is empty
            output_row = entry + ''           # set the entry value as its initial entry
          else:                     # otherwise
            output_row = output_row + ' ' + entry   # add the entry value to the row output
      if s:                         # if s is not empty
        entry_split = entry.split()             # split entry into individual values
        s_entry = s + ' ' + entry_split[0]          # concatenate s and the first value of entry
        string_table.append((new_code, s_entry))      # and append it to the string table with its code
        new_code = new_code + 1               # increment new code
      s = entry + ''                      # copy k to s
    row_output = output_row.split()               # split the output row into individual values
    if not output_image:                    # if output iamge is empty
      output_image = [row_output]               # set the row output as its initial entry
    else:                           # otherwise
      output_image.append(row_output)             # append the row output to the output image
  output_image = np.asarray(output_image)             # convert the output image to a numpy array and reshape it to (height, width)
  return  output_image


# get the arithmetic symbol probabilities from the input key
def get_symbol_probability(input_key, arith_freq_list):
    low = 0.0                             # initialize the low value
    high = 0.0                              # initialize the high value
    for i in range(len(input_key)):                   # traverse the list of keys
        single_key = input_key[i]                   # for each key entry
        value = int(single_key[0])                    # get the value
        probability = float(single_key[1])                # its probability/range
        low = high + 0.0                        # set the current low value to the previous high value
        high = low + probability                    # set the current high value
        if not arith_freq_list:                     # if the arithmetic frequency list is empty
            arith_freq_list = [(value, probability, low, high)]     # set the ovalue's entry as the initial entry
        else:                             # otherwise
            arith_freq_list.append((value, probability, low, high))   # add the value's entry to the arithmetic frequency list
    return arith_freq_list

# Arithmetic decompression algorithm
def arithmetic_decompression(input_image, input_key, arith_freq_list, output_image,width):
    arith_freq_list = get_symbol_probability(input_key, arith_freq_list)      # create the arith freq list from the input key
    newWidth=0
    height = len(input_image)                           # set the image height
    row_code = 0.0                                  # initialize row code
    output_row = []                                 # initialize the output row
    for i in range(len(input_image)):                       # traverse the input image by row
        output_row=[]
        row = input_image[i]                            # extract each row
        row_code = 0.0
        rowstring=row[0]
        newWidth=0
        for h in range(len(rowstring)):                         # traverse the row
            if  rowstring[h]== '1':                           # if the next bit is 1
                x = h + 1                             # adjust the index to start at 1 instead of 0
                row_code = row_code + (1/float(2**x))               # add the frational bit to the row code
        while newWidth < width:                           # assumption for assignment is that image is square: width = height
            for h in range(len(arith_freq_list)):                 # traverse the arith freq list
                current_entry = arith_freq_list[h]                  # get each entry
                if  current_entry[2] <= row_code and row_code < current_entry[3]:       # if the row code float is between the current entry's low and high
                    row_code = (row_code - current_entry[2])/current_entry[1]   # update row code per formula on page 209 of text: (current value - low)/range
                    if not output_row:                      # if the output image is blank
                        output_row = [current_entry[0]]             # set output value as the initial entry
                    else:                             # otherwise
                        output_row.append(current_entry[0])           # add output value to the output image
                    newWidth = newWidth + 1                       # increment the image width
                    break
        if not output_image:
          output_image=[output_row]
        else:
          output_image.append(output_row)
    return  output_image

def create_out_file_no_compression(output_image,output_file_name):
  outfile = open( output_file_name,'w' )
  for line in output_image:
    outfile.write(" ".join(map(str,line))+"\n")
  outfile.flush()
  outfile.close()

def main():
  input_image = []      # initialize input image
  input_key = []        # initialize input key
  symbol_dictionary = []    # initialize symbol_dictionary
  string_table = []     # initialize string table
  arith_freq_list = []    # initialize arithmetic frequency list
  output_image = []     # initialize output image

  if _platform == "linux" or _platform == "linux2":
    slash = '/'
  elif _platform == "darwin":
    slash = '/'
  elif _platform == "win32":
    slash = '\\'

  # Get selection from user input
  selection_code = raw_input("""***File Viewer***\n
  Select one of the following:
  Press 1 for to view a file \n
  Press 2 to exit \n
  Choice: """)

  # view file
  if selection_code == '1':
    # read the quantization file path from user input
    file_dir = raw_input("Enter the path of the file:\n")
    print 'The image file will be read from %s directory' % file_dir
    file_name = raw_input("Enter the image file name:\n")
    full_path = r'{0}{2}{1}'.format(file_dir,file_name,slash)


  #full_path=r'1_1_1_3.tpv'
  #file_name='1_1_1_3.tpv'

  suffix=None
  input_file_name_split=file_name.split('.')
  compression_model_code = get_compression_model(file_name)
  if input_file_name_split[1]=="tpv":
    suffix=".tpy"
  else:
    suffix=".spy"

  output_file_name=input_file_name_split[0]+suffix
  compression_model_code = get_compression_model(file_name)

  #####NEED TO FINISH#####
  #get input data from the input file
  input_image,input_key,width=get_file(full_path, compression_model_code, input_image, input_key)
  #input_image=[item for sublist in input_imageND for item in sublist]
  #columns,rows=np.array(input_imageND).shape
  #print columns
  #print rows
   # No compression
  if compression_model_code == '1':
    output_image = no_compression(input_image, output_image)
    create_out_file_no_compression(output_image,output_file_name)                       # create the output image without any compression
    # Shannon-Fano encoding
  elif compression_model_code == '2':
    output_image = shannon_fano_decompression(input_image, input_key, symbol_dictionary, output_image)    # create the output image using the symbol dictionary
    create_out_file_no_compression(output_image,output_file_name)
  # Dictionary/LZW encoding
  elif compression_model_code == '3':
    output_image = dictionary_lzw_decompression(input_image, input_key, string_table, output_image)     # create the output image using the symbol dictionary
    create_out_file_no_compression(output_image,output_file_name)
  # Arithmetic encoding
  elif compression_model_code == '4':
    output_image = arithmetic_decompression(input_image, input_key, arith_freq_list, output_image,width)      # create the output image code using Arithmetic encoding
    create_out_file_no_compression(output_image,output_file_name)

  else:
    print "Not valid input file"

  #####NEED TO FINISH#####
  # Display the image
  # Exit program
  # if selection_code == 2:
  print 'input file size: {0}'.format(get_file_size(full_path))
  if input_file_name_split[1]=="tpv":
    decodeTPC(output_file_name)
  else:
    spatialDecode(output_file_name)


  origVideo=getFileData(getInputFileName(output_file_name))
  decodeVideo=getFileData(output_file_name+"decoded.txt")

  print 'Original video size: {0}'.format(get_file_size(getInputFileName(output_file_name)))
  print 'Decoded video size: {0}'.format(get_file_size(output_file_name+"decoded.txt"))
  print 'Signal to noise ratio(PSNR): {0}'.format(getPSNR(origVideo, decodeVideo))

def getFileData(filename):
  inFile = open( filename,'r' )
  input_image=[]
  for line in inFile:
    input_image.append([int(float(a)) for a in line.split()])
  return input_image

def getInputFileName(outFileName):
  output_file_name_split=outFileName.split('.')
  if output_file_name_split[1]=="tpy":
    suffix=".spc"
  else:
    suffix=".spc"

  output_file_name_split=outFileName.split('_')
  fileName="{0}_{1}{2}".format(output_file_name_split[0],1,suffix)
  return fileName

def getPSNR(I1, I2):
  I1=np.array(I1)
  I2=np.array(I2)

  s1 = cv2.absdiff(I1,I2)# |I1 - I2|
  s1 = np.float32(s1) #convert to 32Float
  s1 = cv2.multiply(s1, s1) # |I1 - I2|^2

  s = cv2.sumElems(s1)#take the sum of the elements in each channel

  sse = s[0] + s[1] + s[2] # sum channels

  if (sse <= 1e-10):# return 0 if very small number
    return 0
  else:
    mse = sse/(len(I1.shape) * I1.shape[0]*I1.shape[1])#total needs fixing
    psnr = 10 * math.log10(255*255/mse)#Calculate PSNR
    return psnr

def decodeTPC(inputFileName):
  width=10
  height=10
  fileSuffix=".mp4"
  videoDir = raw_input("Enter the video file directory:\n")
  videoFileName=raw_input("Enter the video file name:\n")
  optionNumberList=inputFileName.split('_')
  optionNumber=optionNumberList[1]
  if _platform == "linux" or _platform == "linux2":
    slash = '/'
  elif _platform == "darwin":
    slash = '/'
  elif _platform == "win32":
    slash = '\\'
  fullPath = r'{0}{2}{1}'.format(videoDir,videoFileName+fileSuffix,slash)

  outputFileName=r'{0}_{1}_out{2}'.format(videoFileName,optionNumber,fileSuffix)

  inFile = open( inputFileName,'r' )
  frames= None

  if inputFileName != None:
    if optionNumber=='1':
      frames=tpcDecodingOption1(inFile)
    elif optionNumber=='2':
      frames=tpcDecodingOption2(inFile)
    elif optionNumber=='3':
      frames=tpcDecodingOption3(inFile)
    elif optionNumber=='4':
      frames=tpcDecodingOption4(inFile)
    else:
      print "Input not valid"
      quit()
  else:
    print "Some error while reading video file"

  if frames !=None:
    decodeVideoTPC(frames,fullPath,width,height,outputFileName,inputFileName)

  inFile.flush()
  inFile.close()
# frames: each row represent pixels in a frame which will be reshaped to width and height
def decodeVideoTPC(frames,fullPath,width,height,outputVideoFileName,inputFileName):
  frameSize=frames.shape
  frameRate=30
  fourcc=-1
  print fullPath
  print outputVideoFileName
  cap = cv2.VideoCapture(fullPath)
  outVideoFile=None
  if cap.isOpened:
    if cv2.__version__=='3.0.0':
      frameRate=cap.get(cv2.CAP_PROP_FPS)
      fourcc=cap.get(cv2.CAP_PROP_FOURCC)
    else:
      frameRate=cap.get(cv2.cv.CV_CAP_PROP_FPS)
      fourcc=cap.get(cv2.cv.CV_CAP_PROP_FOURCC)

  fourcc = 828601953
  frameRate=30
  outputFileName=inputFileName+"decoded.txt"
  outfile = open( outputFileName, 'w' )
  #fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
  #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  #fourcc = cv2.VideoWriter_fourcc('I', 'Y', 'U', 'V')
  #fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
  outVideoFile = cv2.VideoWriter(outputVideoFileName, int(fourcc), frameRate,(width,height))
  for x in range(0,frameSize[0]):
      frame=np.array(np.array(frames[x][:]).reshape(width,height), dtype=np.uint8)
      u=np.ones((width,height), dtype=np.uint8)*128
      v=np.ones((width,height), dtype=np.uint8)*128
      yuvImage=cv2.merge((frame,u,v))
      rgbImage = cv2.cvtColor(yuvImage, cv2.COLOR_YUV2BGR)
      cv2.imshow("Decoded Y channel",rgbImage)
      outfile.write(" ".join(map(str,frames[x][:]))+"\n")
      outVideoFile.write(rgbImage)
      c = cv2.waitKey(1)
      if 'q' == chr(c & 255):
        break
  outVideoFile.release()
  cv2.destroyAllWindows()
  print "Output saved to text "+outputFileName
  outfile.flush()
  outfile.close()


def tpcDecodingOption1(inFile):

  lines=None
  for line in inFile:
    if lines==None:
      lines=np.array(list(map(int,line.split())))
    else:
      lines=np.column_stack((lines,list(map(float,line.split()))))
  return lines


def tpcDecodingOption2(inFile):
  lines=None
  for line in inFile:
    encodedSignal = list(map(float,line.split()))
    decodedSignal=[]
    for index,value in enumerate(encodedSignal):
      if index==0:
        decodedSignal.append(value)
      else:
        decodedSignal.append(value+decodedSignal[index-1])

    if lines==None:
      lines=np.array(decodedSignal)

    else:
      lines=np.column_stack((lines,decodedSignal))
  return lines


def tpcDecodingOption3(inFile):
  lines=None
  for line in inFile:
    encodedSignal = list(map(float,line.split()))
    decodedSignal=[]
    for index,value in enumerate(encodedSignal):
      if index<=1:
        decodedSignal.append(int(value))
      else:
        decodedSignal.append(int(value+decodedSignal[index-1]/float(2) + decodedSignal[index-2]/float(2)))

    if lines==None:
      lines=np.array(decodedSignal)

    else:
      lines=np.column_stack((lines,decodedSignal))
  return lines

def tpcDecodingOption4(inFile):
  lines=None
  alpha1=0.5
  alpha2=0.5
  for line in inFile:
    encodedSignal = list(map(float,line.split()))
    decodedSignal=[]
    for k,value in enumerate(encodedSignal):
      if k<=1:
        decodedSignal.append(int(round(encodedSignal[k])))
      else:
        if k<=3:
          alpha1=0.5
          alpha2=0.5
          decodedSignal.append(int(round(value+ alpha1*decodedSignal[k-1]  + alpha2*decodedSignal[k-2])))
        else:
          k1k2Diff = decodedSignal[k-2] - decodedSignal[k-4]
          if k1k2Diff == 0:
            alpha1=0.5
            alpha2=0.5
          else:
            alpha1 = (decodedSignal[k-1] + decodedSignal[k-2]-decodedSignal[k-3]-decodedSignal[k-4])/float(k1k2Diff)
            alpha2 = 1-alpha1

            if alpha1>1 and alpha2>1:
              alpha1=0.5
              alpha2=0.5
            elif alpha1<0 and alpha2<0:
              alpha1=0.5
              alpha2=0.5
            elif alpha1<0 or alpha2>1:
              alpha1=0.0
              alpha2=1.0
            elif alpha1 >1 or alpha2<0:
              alpha1=1.0
              alpha2=0.0
          decodedSignal.append(int(round(value+ alpha1*decodedSignal[k-1]  + alpha2*decodedSignal[k-2])))
    if(len(decodedSignal)>0):
      if lines==None:
        lines=np.array(decodedSignal)

      else:
        lines=np.column_stack((lines,decodedSignal))
  return lines

def spatialPredictiveDecodingOption1(frames, output_file_name):
  #no PC
  outfile = open( output_file_name, 'w' )
  frameCount = len(frames)

  for k in range(0, frameCount):
    for i in range(0, 10):
      for j in range(0, 10):
        outfile.write(str(frames[k][i][j]))
        outfile.write(" ")
    outfile.write("\n")


  newFrames = np.array(frames)

  outfile.flush()
  outfile.close()
  print "\nOutput saved to {0}\n".format(output_file_name)
  return newFrames

def spatialPredictiveDecodingOption2(frames, output_file_name):
  frameCount = len(frames)
  outfile = open( output_file_name, 'w' )

  for k in range(0,frameCount):
    newFrame = []
    for i in range(0, 10):
      newRow = []
      for j in range(0, 10):
        if(j == 0):
          frames[k][i][j] = int(round(frames[k][i][j]))
        else:
          frames[k][i][j] = frames[k][i][j] + frames[k][i][j-1]
          frames[k][i][j] = int(round(frames[k][i][j]))

  for k in range(0, frameCount):
    for i in range (0, 10):
      for j in range(0, 10):
        outfile.write(str(frames[k][i][j]))
        outfile.write(" ")
    outfile.write("\n")


  newFrames = np.array(frames)

  outfile.flush()
  outfile.close()
  print "\nOutput saved to {0}\n".format(output_file_name)
  return newFrames


def spatialPredictiveDecodingOption3(frames, output_file_name):
  frameCount = len(frames)
  current_working_dir = os.getcwd()
  outfile = open( output_file_name, 'w' )

  for k in range(0,frameCount):
    newFrame = []
    for i in range(0, 10):
      newRow = []
      for j in range(0, 10):
        if(i == 0):
          frames[k][i][j] = int(round(frames[k][i][j]))
        else:
          frames[k][i][j] = frames[k][i][j] + frames[k][i-1][j]
          frames[k][i][j] = int(round(frames[k][i][j]))


  for k in range(0, frameCount):
    for i in range (0, 10):
      for j in range(0, 10):
        outfile.write(str(frames[k][i][j]))
        outfile.write(" ")
    outfile.write("\n")

  newFrames = np.array(frames)

  outfile.flush()
  outfile.close()
  print "\nOutput saved to {0}\n".format(output_file_name)
  return newFrames



def spatialPredictiveDecodingOption4(frames, output_file_name):
  frameCount = len(frames)
  current_working_dir = os.getcwd()
  outfile = open( output_file_name, 'w' )

  for k in range(0,frameCount):
    newFrame = []
    for i in range(0, 10):
      newRow = []
      for j in range(0, 10):
        if i == 0 or j == 0:
          if i == 0 and j != 0:
            frames[k][i][j] = int(round(frames[k][0][j]))
          elif i != 0 and j == 0:
            frames[k][i][j] = int(round(frames[k][i][0]))
          elif i == 0 and j == 0:
            frames[k][i][j] = int(round(frames[k][0][0]))
        else:
          frames[k][i][j] = frames[k][i][j] + frames[k][i-1][j-1]
          frames[k][i][j] = int(round(frames[k][i][j]))

  for k in range(0, frameCount):
    for i in range (0, 10):
      for j in range(0, 10):
        outfile.write(str(frames[k][i][j]))
        outfile.write(" ")
    outfile.write("\n")


  newFrames = np.array(frames)

  outfile.flush()
  outfile.close()
  print "\nOutput saved to {0}\n".format(output_file_name)
  return newFrames

def spatialPredictiveDecodingOption5(frames, output_file_name):
  frameCount = len(frames)
  current_working_dir = os.getcwd()
  outfile = open( output_file_name, 'w' )

  for k in range(0, frameCount):
    for i in range(0, 10):
      for j in range(0, 10):
        if i == 0 or j == 0:
          if i == 0 and j != 0:
            frames[k][i][j] = int(round(frames[k][i][j]))
          elif i != 0 and j == 0:
            frames[k][i][j] = int(round(frames[k][i][j]))
          elif i == 0 and j == 0:
            frames[k][i][j] = int(round(frames[k][i][j]))
        elif j == 1 or j == 2 or j == 3:
          alpha1 = 0.33
          alpha2 = 0.33
          alpha3 = 0.33

          predictorA = frames[k][i][j-1] * alpha1
          predictorB = frames[k][i-1][j] * alpha2
          predictorC = frames[k][i-1][j-1] * alpha3
          totalPredictor = predictorA + predictorB + predictorC

          frames[k][i][j] = frames[k][i][j] + totalPredictor
          frames[k][i][j] = int(round(frames[k][i][j]))

        else:
          predictedValOne = frames[k][i][j-3]
          predictedValOneA = frames[k][i][j-4]
          predictedValOneB = frames[k][i-1][j-3]
          predictedValOneC = frames[k][i-1][j-4]

          predictedValTwo = frames[k][i][j-2]
          predictedValTwoA = frames[k][i][j-3]
          predictedValTwoB = frames[k][i-1][j-2]
          predictedValTwoC = frames[k][i-1][j-3]

          predictedValThree = frames[k][i][j-1]
          predictedValThreeA = frames[k][i][j-2]
          predictedValThreeB = frames[k][i-1][j-1]
          predictedValThreeC = frames[k][i-1][j-2]

          a = np.array([[predictedValOneA, predictedValOneB, predictedValOneC],[predictedValTwoA, predictedValTwoB, predictedValTwoC],[predictedValThreeA, predictedValThreeB, predictedValThreeC]])

          b = np.array([predictedValOne, predictedValTwo, predictedValThree])

          if np.linalg.cond(a) < 1/sys.float_info.epsilon:
            x = np.linalg.solve(a, b)
            alpha1 = x[0]
            alpha2 = x[1]
            alpha3 = x[2]

            if alpha1 >= 0.0 and alpha1 < 1.0:
              if alpha2 >= 0.0 and alpha2 < 1.0:
                if alpha3 >= 0.0 and alpha3 < 1.0:
                  sumAlpha = alpha1 + alpha2 + alpha3

                  if sumAlpha != 1.0:
                    alpha1 = 0.33
                    alpha2 = 0.33
                    alpha3 = 0.33
                else:
                  alpha1 = 0.33
                  alpha2 = 0.33
                  alpha3 = 0.33
              else:
                alpha1 = 0.33
                alpha2 = 0.33
                alpha3 = 0.33
            else:
              alpha1 = 0.33
              alpha2 = 0.33
              alpha3 = 0.33
          else:
            alpha1 = 0.33
            alpha2 = 0.33
            alpha3 = 0.33

          predictorA = frames[k][i][j-1] * alpha1
          predictorB = frames[k][i-1][j] * alpha2
          predictorC = frames[k][i-1][j-1] * alpha3

          totalPredictor = predictorA + predictorB + predictorC

          frames[k][i][j] = frames[k][i][j] + totalPredictor
          frames[k][i][j] = int(round(frames[k][i][j]))


  for k in range(0, frameCount):
    for i in range (0, 10):
      for j in range(0, 10):
        outfile.write(str(frames[k][i][j]))
        outfile.write(" ")
    outfile.write("\n")

  newFrames = np.array(frames)

  outfile.flush()
  outfile.close()
  print "\nOutput saved to {0}\n".format(output_file_name)
  return newFrames


def decodeVideoSPC(frames,fullPath,width,height,outputVideoFileName):
  size = len(frames)
  frameRate=30
  fourcc=-1

  cap = cv2.VideoCapture(fullPath)
  outVideoFile=None
  if cap.isOpened:
    if cv2.__version__=='3.0.0':
      frameRate=cap.get(cv2.CAP_PROP_FPS)
      fourcc=cap.get(cv2.CAP_PROP_FOURCC)
    else:
      frameRate=cap.get(cv2.cv.CV_CAP_PROP_FPS)
      fourcc=cap.get(cv2.cv.CV_CAP_PROP_FOURCC)

  #fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
  #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  #fourcc = cv2.VideoWriter_fourcc('I', 'Y', 'U', 'V')
  fourcc = 828601953
  frameRate=30
  outVideoFile = cv2.VideoWriter(outputVideoFileName, int(fourcc), frameRate,(width,height))
  for x in range(0,size):
      frame=np.array(frames[x], dtype=np.uint8)
      u=np.ones((width,height), dtype=np.uint8)*128
      v=np.ones((width,height), dtype=np.uint8)*128
      yuvImage=cv2.merge((frame,u,v))
      rgbImage = cv2.cvtColor(yuvImage, cv2.COLOR_YUV2BGR)
      cv2.imshow("Decoded Y channel",rgbImage)
      outVideoFile.write(rgbImage)
      c = cv2.waitKey(1)
      if 'q' == chr(c & 255):
        break
  outVideoFile.release()
  cv2.destroyAllWindows()



def spatialDecode(inputFileName):

  width=10
  height=10
  fileSuffix=".mp4"
  videoDir = raw_input("Enter the video file directory:\n")
  videoFileName=raw_input("Enter the video file name:\n")
  optionNumberList=inputFileName.split('_')
  optionNumber=optionNumberList[1]
  if _platform == "linux" or _platform == "linux2":
    slash = '/'
  elif _platform == "darwin":
    slash = '/'
  elif _platform == "win32":
    slash = '\\'
  fullPath = r'{0}{2}{1}'.format(videoDir,videoFileName+fileSuffix,slash)

  outputFileName=r'{0}_{1}_out{2}'.format(videoFileName,optionNumber,fileSuffix)

  frames= None

  txt_file_name =optionNumberList[0]

  option_number=optionNumber

  output_file_name=inputFileName+"decoded.txt"

  count = 0

  Frames = []

  with open(inputFileName) as openfileobject:
    for line in openfileobject:
      frame = []
      for char in line.split():
        number = float(char)
        frame.append(number)
        count = count + 1
        #print number
      Frames.append(frame)

  numFrames = count/100

  newFrames = []

  for k in range(0, numFrames):
    frame = Frames[k]
    newframe = []
    for i in range(0, 10):
      row = []
      multiplier = i*10
      for j in range(0, 10):
        row.append(frame[multiplier+j])
      newframe.append(row)
    newFrames.append(newframe)

  finalFrames = []

  if option_number == '1':
    finalFrames = spatialPredictiveDecodingOption1(newFrames, output_file_name)
  elif option_number == '2':
    finalFrames = spatialPredictiveDecodingOption2(newFrames, output_file_name)
  elif option_number == '3':
    finalFrames = spatialPredictiveDecodingOption3(newFrames, output_file_name)
  elif option_number == '4':
    finalFrames = spatialPredictiveDecodingOption4(newFrames, output_file_name)
  elif option_number == '5':
    finalFrames = spatialPredictiveDecodingOption5(newFrames, output_file_name)
  else:
    print "Input not valid"
    quit()

  if finalFrames != None:
    decodeVideoSPC(finalFrames,fullPath,width,height,outputFileName)

main()