import sys
import cv2
import numpy as np
import os
import lzw
import arcode
import collections
import math
import struct
from decimal import Decimal


def get_file_size(path):
  fileHandle = open(path, 'rb')
  byteArr = bytearray(fileHandle.read(os.path.getsize(path)))
  fileHandle.close()
  fileSize = len(byteArr)
  return fileSize

# Used to show input prompt component names
def compression_model_components(compression_model_code) :
  if compression_model_code == '1':
    compression_model = "no compression"
  elif compression_model_code == '2':
    compression_model = "Shannon-Fano"
  elif compression_model_code == '5':
    compression_model = "Dictionary/LZW"
  elif compression_model_code == '6':
    compression_model = "Arithmetic"
  elif compression_model_code == '3':
    compression_model = "LZW"
  elif compression_model_code == '4':
    compression_model = "Arithmetic"
  else :
    print 'Not a valid compression model. Rerun program and choose a selection between 1 to 6\n'
    sys.exit(1)
  return compression_model

#####NEED TO FINISH#####
# gets the video data contained in the quantized file  - need to finish once have info about task 3 output
def get_file(full_path,compression_model_code):
  inputList=[]
  inFile = open( full_path )
  print "Input file size {0}".format(get_file_size(full_path))
  if compression_model_code == '1':
    for line in inFile:
      inputList.append([float(a) for a in line.split()])
      #inputList.append(list(map(float,line.split())))
  else:
    for line in inFile:
      inputList.append([int(float(a)) for a in line.split()])

  return np.array(inputList)

#####NEED TO FINISH#####
# create the output file from the symbol dictionary and output image or output image code
#def create_output_file(compression_model_code, symbol_dictionary, arith_freq_list, string_table, output_image, file_name):

# convert the image to binary without any compression
def no_compression(input_image, output_image):
  #pixel = ''                            # initialize variable for binary conversion
  #for i in np.nditer(input_image):                # traverse the image array and
  #  pixel = bin(((1 << 8) - 1) & -3)[2:]                      # convert each value in the image to binary
    #pixel =floatToRawLongBits(i)
  #  output_image = np.append(output_image, pixel.zfill(8))    # padd it to 8 digits and add to the output image
  #output_image = np.reshape(output_image, input_image.shape)    # make the output array the same shape as the input array
  return input_image


# create the symbol dictionary using frequency count of the values in the image
def create_symbol_dictionary(input_image, symbol_dictionary):
  #frequency_list = [0]*256                      # create blank list of all possible values
  #for i in np.nditer(input_image):                  # go through the image
  #  frequency_list[i] +=1                     # and obtain a count for each value in the image
  frequency_list=collections.Counter(np.reshape(input_image, np.product(input_image.shape)))
  for idx,value in enumerate(frequency_list.keys()):                        # put all of the values with a count into the dictionary
    freq_val = frequency_list[value]
    if freq_val > 0:                       # with their respective count
      if not symbol_dictionary:                 # if the symbol dictionary is empty
        symbol_dictionary = [(value, freq_val, '')]    # set the current output as its initial entry
      else:                           # otherwise
        symbol_dictionary.append((value, freq_val, ''))  # and a blank place holder for their compression symbol
  return symbol_dictionary

# create the output image using the Shannon-Fanno Symbol Dictionary
def create_output_image_shannon(input_image, output_image, symbol_dictionary):
  output_image = np.copy(input_image)               # copy the input image to the output image
  output_image = output_image.astype('str')           # change the output image to an array of strings
  for i in range(len(symbol_dictionary)):             # go through the dictionary
    symbol = symbol_dictionary[i]               # for each entry in the dictionary
    for h in np.nditer(output_image, op_flags=['readwrite']): # traverse the output image as read/writeable
      if int(h.tolist()) == symbol[0]:            # for each value in the image that matches the dictionary entry
        h[...] = symbol[2]                  # change the value to its compression symbol
  return output_image                       # and return the compressed imagev

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

# Shanon-Fano compression algorithm
def shannon_fano_compression(input_image, symbol_dictionary, output_image):
  symbol_dictionary = create_symbol_dictionary(input_image, symbol_dictionary)        # create the symbol_dictionary
  symbol_dictionary.sort(key=lambda symbol_dictionary: symbol_dictionary[1], reverse=True)  # sort symbol_dictionary in descending order
  top_index = 0                                       # initial first index
  bottom_index = len(symbol_dictionary) - 1                         # initial last index
  symbol_dictionary = shannon_fano_algorithm(symbol_dictionary, top_index, bottom_index)    # call recursive shannon-fano algorithm
  output_image = create_output_image_shannon(input_image, output_image, symbol_dictionary)      # create the output_image
  return  output_image,symbol_dictionary


# convert the LZW/Dictionary image to binary
def binary_conversion(output_image):
  binary_image = []                       # initialize the empty binary image
  binary_row = []                         # initialize theempty binary row
  pixel = ''                            # initialize variable for binary conversion
  for i in range(len(output_image)):                # traverse the image array
    row = output_image[i]                   # get each row
    binary_row = []                       # reset the binary row
    for j in range(len(row)):                 # traverse each row
      pixel = bin(row[j])[2:]                 # convert each value in the image to binary
      if not binary_row:                    # if the binary row is empty
        binary_row = [pixel.zfill(8)]           # set the pixel as its first value
      else:                         # otherwise
        binary_row.append(pixel.zfill(8))         # pad it to 8 digits and add it to the binary row
    if not binary_image:                    # if the binary image is empty
      binary_image = [binary_row]               # set the binary row as its first value
    else:                           # otherwise
      binary_image.append(binary_row)             # add it to the output image
  output_image = list(binary_image)               # copy to binary image to the output image
  return output_image

# Create LZW String Table
def create_string_table(input_image, symbol_dictionary, string_table):
  symbol_dictionary = create_symbol_dictionary(input_image, symbol_dictionary)  # create the
  code = 1                                    # initialize the code counter
  for i in range(len(symbol_dictionary)):                     # traverse the symbol dictionary
    row = symbol_dictionary[i]                          # get each entry
    if not string_table:                            # if the string table is empty
      string_table = [(code, str(row[0]))]                  # set the current output as its initial entry
    else:                                   # otherwise
      string_table.append((code, str(row[0])))                # add the current output to the table
    code = code + 1                               # increment the code counter
  return string_table                       # format: s, c, output, code, string

# Dictionary/LZW compression algorithm
def create_output_image(input_image, symbol_dictionary, string_table, output_image):
  s = ''                                        # initialize s
  c = ''                                        # initialize c
  sc = ''                                       # initialize sc
  found = 0                                     # initialize found to 0
  string_table = create_string_table(input_image, symbol_dictionary, string_table)  # create the string table
  new_code = len(string_table)+1                  # new code is one more than the length of the current string table
  height, width = input_image.shape                         # get the shape of the input image
  for i in range(height):                               # traverse the image array row by row
    row = input_image[i,:]                              # and extract each row
    row_code = []                                 # reset the row code to blank
    s = str(row[0])                                 # set s to the initial value
    for j in range(1, width):                           # traverse the rest of the current row
      c = str(row[j])                               # c = next value
      sc = s + ' ' + c                              # sc = s + c
      for k in range(len(string_table)):                      # traverse the string table
        string_row = string_table[k]                      # get the current entry from the string table
        if string_row[1] == sc:                         # if sc already exists in the string table
          s = sc + ''                             # replace s with sc
          found = 1                             # and set found to 1
          break
      if found == 0:                                # if found == 0 (sc was not found)
        for l in range(len(string_table)):                    # traverse the string table
          string_row2 = string_table[l]                   # get the current entry
          if string_row2[1] == s:                       # if s already exists in the string table
            if not row_code:                        # and if row code is empty
              row_code = [string_row2[0]]                 # set the code for s as the initial value in the output code
            else:                             # otherwise
              row_code.append(string_row2[0])               # add the code for s to the output code with a space
            break
        string_table.append((new_code, sc))                   # add sc to the string table
        s = c + ''                                # replace the value of s with the value of c (advance to the next pixel)
        new_code = new_code + 1                         # increment new code
      found = 0                                 # reset found to 0
    for m in range(len(string_table)):                        # travers the string table
      string_row3 = string_table[m]                       # for each entry
      if string_row3[1] == s:                           # if the entry's string matches s
        row_code.append(string_row3[0])                     # add the code for s to the row code
        break
    if not output_image:                              # if the output image is empty
      output_image = [row_code]                     # set the row code as its initial entry
    else:                                     # otherwise
      output_image.append(row_code)                     # append the row code to the output image
  output_image = binary_conversion(output_image)                    # convert the output image to binary
  return  output_image,string_table


# get the symbol probabilities, ranges, and starting highs & lows for Arithmetic encoding
def get_symbol_frequency(input_image, symbol_dictionary, arith_freq_list):
  s_sum = 0                                     # initialize the sum variable
  low = 0.0                                     # initialize the range low
  high = 0.0                                      # initialize the range high
  s_range = 0.0                                   # initialize the symbol's range
  symbol_dictionary = create_symbol_dictionary(input_image, symbol_dictionary)    # create a symbol dictionary to get the counts of each value in the image
  size = len(symbol_dictionary)                           # get the length of the dictionary
  for i in range(size):                               # go through the symbol dictionary
    symbol = symbol_dictionary[i]                         # and for each entry
    s_sum = s_sum + symbol[1]                           # add its count to the sum
  for i in range(size):                               # go back through the symbol dictionary
    symbol = symbol_dictionary[i]                         # and for each entry:
    low = high + 0.0                                # current range low is equal to the previous range's high
    s_range = symbol[1]/float(s_sum)                        # current range = symbol's probability
    high = low + s_range                              # current high = current low + current range
    if not arith_freq_list:                             # if the arithmetic frequency list is empty
      arith_freq_list = [(symbol[0], s_range, low, high)]             # set the symbol, its probability, its range low, and it range high as the initial entry
    else:                                     # otherwise
      arith_freq_list.append((symbol[0], s_range, low, high)) # apend the symbol, its probability, its range low, and its range high to the arithmetic frequency list
  return arith_freq_list

# Arithmetic compression algorithm
def arithmetic_compression(input_image, symbol_dictionary, arith_freq_list, output_image):
  arith_freq_list = get_symbol_frequency(input_image, symbol_dictionary, arith_freq_list)   # get each symbol probabilities, starting range low and starting range high
  height, width = input_image.shape                             # get the shape of the input image
  range_low = 0.0                                          # initialize range low
  range_high = 0.0                                         # initialize range high
  for i in range(height):                                   # traverse the image array row by row
    low = 0.0                                       # reset low variable
    high = 1.0                                        # reset high variable
    s_range = 1.0                                     # reset range variable
    row_code = 0.0                                     # reset the row code
    row_binary = ''                                     # reset the row's binary representation
    r = 1                                         # reset row code build counter
    row = input_image[i,:]                                  # and extract each row
    for h in range(width):                                  # traverse the row
      symbol = row[h]                                   # get each symbol
      for j in range(len(arith_freq_list)):                       # search the frequency list
        current_symbol = arith_freq_list[j]                       # check for the current symbol
        if symbol == current_symbol[0]:                         # when you find it
          range_low = current_symbol[2]                       # get the range low
          range_high = current_symbol[3]                        # and the range high
      s_range = high - low
      high = low + s_range * range_high                         # create the new low, high and range -
      low = low + s_range * range_low                           #   formulas from page 206
                                      #   of the textbook
    while row_code < low:                                 # while row code value is smaller than the final range low
      current_try = row_code + (1/float(2**r))                      # add the rth binary fractional bit
      if current_try > high:                                # if it makes the value higher than the range high
        row_binary = row_binary + '0'                         # replace the rth bit with 0
      else:                                       # otherwise
        row_binary = row_binary + '1'                         # assign 1 to rth bit
        row_code = current_try                              # and keep the new row code value
      r = r + 1                                     # increment to counter
    if not output_image:                                  # if the output image is empty
      output_image = [row_binary]                             # set the row binary as its initial value
    else:                                         # otherwise
      output_image.append(row_binary)                           # append the arithmetic comrepssion code for the row to the image
  return  output_image, arith_freq_list

def create_output_file_no_compression(input_image,output_file_name):
  outfile = open( output_file_name,'wb' )

  for i in input_image.tolist():
    outfile.write(" ".join(map(str,i))+"\n")

  outfile.flush()
  outfile.close()

def create_output_file_shannon_fano(symbol_dictionary,output_image,output_file_name):
  outfile = open( output_file_name,'wb' )
  input_key=[]
  for symbol in symbol_dictionary:
    outfile.write(" ".join(map(str,symbol[:2]))+",")
    #input_key.append(symbol[:2])
  outfile.write("\n")
  for i in output_image.tolist():
    outfile.write(" ".join(map(str,i))+"\n")

  outfile.flush()
  outfile.close()

def create_output_file_lzw(string_table, output_image, output_file_name):
  outfile = open( output_file_name,'wb' )
  input_key=[]
  for symbol in string_table:
    outfile.write(" ".join(map(str,symbol[:2]))+",")
    #input_key.append(symbol[:2])
  outfile.write("\n")
  for i in output_image:
    outfile.write(" ".join(map(str,i))+"\n")

  outfile.flush()
  outfile.close()

def create_output_file_arithmetic(arith_freq_list, output_image, output_file_name,input_image):
  outfile = open( output_file_name,'wb' )
  input_key=[]
  for symbol in arith_freq_list:
    outfile.write(" ".join(map(str,symbol[:2]))+",")
    #input_key.append(symbol[:2])
  outfile.write("\n")
  height, width = input_image.shape
  outfile.write(str(width)+"\n")
  for i in output_image:
    outfile.write(str(i)+"\n")

  outfile.flush()
  outfile.close()


def main():

  symbol_dictionary = []
  string_table = []
  arith_freq_list = []
  output_image = []
  output_key = []

  # read the quantization file path from user input
  file_dir = raw_input("Enter the path of the error quantization file:\n")
  print 'The error quantization file will be read from %s directory' % file_dir
  file_name = raw_input("Enter the error quantization file name:\n")
  full_path = r'{0}\{1}'.format(file_dir,file_name)


  # Read the compression mode from user input
  compression_model_code = raw_input("""Select a Compression Model:\n
  Press 1 for no compression \n
  Press 2 for Variable-length encoding with Shannon-Fano coding \n
  Press 3 for Dictionary encoding with LZW coding \n
  Press 4 for Arithmetic coding \n
  Model: """)
  #full_path=r'1_1_1.tpq'
  #file_name='1_1_1.tpq'
  #compression_model_code='3'

  suffix=None
  input_file_name_split=file_name.split('.')

  if input_file_name_split[1]=="tpq":
    suffix=compression_model_code+".tpv"
  else:
    suffix=compression_model_code+".spv"

  output_file_name=input_file_name_split[0]+"_"+suffix


  compression_model = compression_model_components(compression_model_code)
  print "You have selected the following compression model: " + compression_model + "\n"
  input_image = get_file(full_path,compression_model_code)
  #input_image=np.array([[1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2,1,2,3,1,1,1,1,2,2,2]])

  # no compression
  if compression_model_code == '1':
    output_image = no_compression(input_image, output_image)
    create_output_file_no_compression(output_image,output_file_name)                                 # create the output image without any compression
    #create_output_file(compression_model_code, symbol_dictionary, arith_freq_list, string_table, output_image, file_name)    # create the output file
  # Shannon-Fano encoding
  if compression_model_code == '2':
    global fileSize
    inputFile = full_path
    outputFile = output_file_name

    # read the whole input file into a byte array
    fileSize = os.path.getsize(inputFile)
    fi = open(inputFile, 'rb')
    # byteArr = map(ord, fi.read(fileSize))
    byteArr = bytearray(fi.read(fileSize))
    fi.close()
    fileSize = len(byteArr)

    output_image, symbol_dictionary = shannon_fano_compression(input_image, symbol_dictionary, output_image)          # create the output image using the symbol dictionary
    create_output_file_shannon_fano(symbol_dictionary,output_image,output_file_name)
    #create_output_file(compression_model_code, symbol_dictionary, arith_freq_list, string_table, output_image, file_name)    # create the output file
  # Dictionary/LZW encoding
  if compression_model_code == '5':
    output_image, string_table = create_output_image(input_image, symbol_dictionary, string_table, output_image)        # create the output image using the symbol dictionary
    create_output_file_lzw(string_table, output_image, output_file_name)   # create the output file
  # Arithmetic encoding
  if compression_model_code == '6':
    output_image, arith_freq_list = arithmetic_compression(input_image, symbol_dictionary, arith_freq_list, output_image)   # create the output image code using Arithmetic encoding
    create_output_file_arithmetic(arith_freq_list, output_image, output_file_name,input_image)
    #create_output_file(compression_model_code, symbol_dictionary, arith_freq_list, string_table, output_image, file_name)   # create the output file
  if compression_model_code == '3':
    lzw.writebytes(output_file_name, lzw.compress(b"".join(lzw.readbytes(full_path))))
  if compression_model_code == '4':
    ar = arcode.ArithmeticCode(False)
    ar.encode_file(full_path, output_file_name)

  print "Output file size {0}".format(get_file_size(output_file_name))

main()