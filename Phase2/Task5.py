import sys
import cv2
import numpy as np

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

#####NEED TO FINISH#####
# gets the data contained in the compression file  - need to finish once have info about task 3 output
def get_file(full_path, compression_model_code, input_image, input_key):
  #no compression input image - input_image = np.array([], str)
  # Shannon_fano input image - input_image = np.array([], str)
  # LZW/Dictionary input image - input_image = []
  # Arithmetic input image - input_image = []
  inFile = open( full_path )
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
  temp=[]
  for i,value in enumerate(input_image):               # traverse the image array and
    temp = [int(x, 2) for x in value]                 # convert each binary value in the image to an integer
    if not output_image:
      output_image=[temp]
    else:
      output_image.append(temp)               # and add to the output image
  output_image=np.asarray(output_image,dtype=np.uint8)
  return output_image


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
      symbol_dictionary = [(int(single_key[0]), int(single_key[1]), '')]      # set the key entry as its initial value and a blank place holder for their compression symbol
    else:                                     # otherwise
      symbol_dictionary.append((int(single_key[0]), int(single_key[1]), ''))    # add the key entry to the dictionary with a blank place holder for their compression symbol
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

  '''
  # Get selection from user input
  selection_code = raw_input("""***File Viewer***\n
  Select one of the following:
  Press 1 for to view a file \n
  Press 2 to exit \n
  Choice: """)

  # view file
  if selection_code == 1:
    # read the quantization file path from user input
    file_dir = raw_input("Enter the path of the file:\n")
    print 'The image file will be read from %s directory' % file_dir
    file_name = raw_input("Enter the image file name:\n")
    full_path = r'{0}\{1}'.format(file_dir,file_name)
  '''

  full_path=r'1_1_1_3.tpv'
  file_name='1_1_1_3.tpv'

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
  # Arithmetic encoding
  elif compression_model_code == '4':
    output_image = arithmetic_decompression(input_image, input_key, arith_freq_list, output_image,width)      # create the output image code using Arithmetic encoding
  else:
    print "Not valid input file"
  print output_image
  #####NEED TO FINISH#####
  # Display the image
  # Exit program
  # if selection_code == 2:

main()