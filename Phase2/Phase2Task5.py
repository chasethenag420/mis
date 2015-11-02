import sys
import cv
import numpy as np
import helpers as hl

#####NEED TO FINISH#####
# get the compression model from the file name
def get_compression_model(file_name, compression_model_code):
	return compression_model_code

#####NEED TO FINISH#####
# gets the data contained in the compression file  - need to finish once have info about task 3 output
def get_file(full_path, file_name, compression_model_code, input_image, input_key):
	#no compression input image - input_image = np.array([], str)
	# Shannon_fano input image - input_image = np.array([], str)
	# LZW/Dictionary input image - input_image = []
	# Arithmetic input image - input_image = []

	return input_image, input_key


# convert the image from binary without any compression
def no_compression(input_image, output_image):
	for i in np.nditer(input_image):								# traverse the image array and
 		temp = int(i.tolist(), 2)									# convert each binary value in the image to an integer
		output_image = np.append(temp)								# and add to the output image
	output_image = np.reshape(output_image, input_image.shape)		# make the output array the same shape as the input array
	return output_image


# create the output image using the symbol_dictionary
def create_output_image(input_image, output_image, symbol_dictionary):
	output_image = np.copy(input_image)								# copy the input image to the output image - it is an array of strings at this point
	for i in range(len(symbol_dictionary)):							# go through the dictionary
		symbol = symbol_dictionary[i]								# for each entry in the dictionary
		for h in np.nditer(output_image, op_flags=['readwrite']):	# traverse the output image as read/writeable
			if h == symbol[2]:										# for each compression symbol in the image that matches the dictionary entry
				h[...] = str(symbol[0])								# change the compression symbol to its represented value
	output_image = output_image.astype('int')						# change the output image to an array of integers
	return output_image 											# and return the decompressed image

# get the symbol counts from the input key
def create_symbol_dictionary(input_key, symbol_dictionary):
	for i in range(len(input_key)):														# traverse the input key 
		single_key = input_key[i]														# get each key entry
		if not symbol_dictionary:														# if the symbol frequency list is empty
			symbol_dictionary = [(int(single_key[0]), int(single_key[1]), '')]			# set the key entry as its initial value and a blank place holder for their compression symbol
		else:																			# otherwise
			symbol_dictionary.append((int(single_key[0]), int(single_key[1]), ''))		# add the key entry to the dictionary with a blank place holder for their compression symbol
	return symbol_dictionary

# Recursive Shannon-Fano algorithm
def shannon_fano_algorithm(symbol_dictionary, top_index, bottom_index):
	s_avg = 0																		# initialize current range average
	s_mid = 0																		# initialize current range midpoint
	s_sum = 0																		# initialize cvurrent range sum
	split = 0																		# initialize current range split point
	size = bottom_index - top_index + 1												# set current range size
	if size > 1:																	# while there are entries in the dictionary
		for i in range(top_index, bottom_index + 1):								# loop through
			symbol = symbol_dictionary[i]											# the symbol dictionary to 
			s_sum = s_sum + symbol[1]												# get the sum of the current range
		s_avg = int(s_sum / size)													# determine the average of the range
		s_mid = int(s_sum / 2 + top_index)											# determine the mid point of the range
		split = s_mid + s_avg														# determine the split point to divide the range into 2 groups
		for i in range(top_index, bottom_index + 1):								# for loop through
			symbol = symbol_dictionary[i]											# symbol dictionary to build tree
			if i < split:															# for the left branch of tree
				symbol_dictionary[i] = (symbol[0], symbol[1], symbol[2] + '0')		# add next digit to the left branch's symbol
			else:																	# and for the right branch of tree
				symbol_dictionary[i] = (symbol[0], symbol[1], symbol[2] + '1')		# add next digit to the right branch's symbol
		shannon_fano_algorithm(symbol_dictionary, top_index, split - 1)				# recursive call for the left branch
		shannon_fano_algorithm(symbol_dictionary, split, bottom_index)				# recursive call for the right branch
	return symbol_dictionary

# Shanon-Fano decompression algorithm
def shannon_fano_decompression(input_image, input_key, symbol_dictionary, output_image):
	symbol_dictionary = create_symbol_dictionary(input_key, symbol_dictionary)					# create the symbol_dictionary
	top_index = 0																				# initial first index
	bottom_index = len(symbol_dictionary) - 1													# initial last index
	symbol_dictionary = shannon_fano_algorithm(symbol_dictionary, top_index, bottom_index)		# call recursive shannon-fano algorithm to build compression symbols
	output_image = create_output_image(input_image, output_image, symbol_dictionary)			# create the output image
	return  output_image


# convert the LZW/Dictionary image from binary to integers
def convert_from_binary(input_image):
	integer_image = []								# initialize the empty integer image
	for i in range(len(input_image)):				# traverse the image array
		row = input_image[i]						# get each row
		integer_row = []							# reset the integer_row to empty
		for j in range(len(row)):					# traverse the row
			value = int(row[j], 2)					# convert each value in the row to an integer
			if not integer_row:						# if the integer row is empty
				integer_row = [value]				# set the value as its initial entry
			else:									# otherwise
				integer_row.append(valeu)			# append it to the integer row
		if not integer_image:						# if the integer image is empty
			integer_image = [integer_row]			# set the integer row as it initial value
		else:										# otherwise
			integer_image.append(integer_row)		# append the integer row to the integer image
	input_image = list(integer_image)				# copy the integer image to the output image
	return input_image

# Create the string table from the input key
def create_string_table(input_key, string_table):
	for i in range(len(input_key)):												# traverse the input key 
		single_key = input_key[i]												# get eack key entry
		if not string_table:													# if the symbol frequency list is empty
			string_table = [(int(single_key[0]), str(single_key[1]))]			# set the key entry as its initial value and a blank place holder for their compression symbol
		else:																	# otherwise
			string_table.append((int(single_key[0]), str(single_key[1])))		# add the key entry to the dictionary with a blank place holder for their compression symbol
	return symbol_dictionary

#####NEED TO FINISH#####
# Dictionary/LZW decompression algorithm
def dictionary_lzw_decompression(input_image, input_key, string_table, output_image):
	string_table = create_string_table(input_key, string_table)						# create the string table for the input key
	input_image = convert_from_binary(input_image)									# convert the input image from binary to integers
	s = ''																			# initialize s to nil
	for i in range(len(input_image))												# traverse the input image
		row = input_image[i]														# get each row from the image
		row_output = []																# reset row output to empty
		for j in range(len(row)):													# traverse the row
			input_code_k = row[j]													# get each input code
			for k in range(len(string_table)):										# traverse the string table
				row_entry = string_table[k]											# get each entry in the table
				if row_entry[0] == input_code_k:									# if the input code matches the entry in the string table
					if not row_output:												# if 
					row_output = [input_code_k]


	output_image = np.asarray(output_image)											# convert the output image to a numpy array and reshape it to (height, width)
	return  output_image


# get the arithmetic symbol probabilities from the input key
def get_symbol_probability(input_key, arith_freq_list):
	low = 0.0															# initialize the low value
	high = 0.0															# initialize the high value
	for i in range(len(input_key)):										# traverse the list of keys 
		single_key = input_key[i]										# for each key entry
		value = int(single_key[0])										# get the value
		probability = float(single_key[1])								# its probability/range
		low = high + 0.0												# set the current low value to the previous high value
		high = low + probability										# set the current high value
		if not arith_freq_list:											# if the arithmetic frequency list is empty
			arith_freq_list = [(value, probability, low, high)]			# set the ovalue's entry as the initial entry
		else:															# otherwise
			arith_freq_list.append((value, probability, low, high))		# add the value's entry to the arithmetic frequency list
	return arith_freq_list

# Arithmetic decompression algorithm
def arithmetic_decompression(input_image, input_key, arith_freq_list, output_image):
	arith_freq_list = get_symbol_probability(input_key, arith_freq_list)			# create the arith freq list from the input key
	output_image = np.asarray														# convert the output image to a numpy aray
	width = 0																		# initialize the image width value
	height = len(input_image)														# set the image height
	row_code = 0.0																	# initialize row code
	output_row = []																	# initialize the output row
	for i in range(len(input_image)):												# traverse the input image by row
		row = input_image[i]														# extract each row
		for h in range(len(row)):													# traverse the row
			if row[h] == '1':														# if the next bit is 1
				x = h + 1															# adjust the index to start at 1 instead of 0
				row_code = row_code + (1/(float(2**x)))								# add the frational bit to the row code
		while width < height:														# assumption for assignment is that image is square: width = height
			for h in range(len(arith_freq_list)):									# traverse the arith freq list
				current_entry = arith_freq_list[h]									# get each entry
				if  current_entry[2] <= row_code <= current_entry[3]:				# if the row code float is between the current entry's low and high
					row_code = (row_code - current_entry[2])/current_entry[1]		# update row code per formula on page 209 of text: (current value - low)/range
					if not output_image:											# if the output image is blank
						output_image = [current_entry[0]]							# set output value as the initial entry
					else:															# otherwise
						output_image.append(current_entry[0])						# add output value to the output image
					width = width + 1												# increment the image width
	output_image = np.reshape(np.asarray(output_image), (height, width))			# convert the output image to a numpy array and reshape it to (height, width)
	return  output_image


def main():
	input_image = []			# initialize input image
	input_key = []				# initialize input key
	symbol_dictionary = []		# initialize symbol_dictionary
	string_table = []			# initialize string table
	arith_freq_list = []		# initialize arithmetic frequency list
	output_image = []			# initialize output image

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
		current_working_dir = os.getcwd()

		#####NEED TO FINISH#####
		#get input data from the input file

		compression_model_code = get_compression_model(file_name, compression_model_code)
		# No compression
		if compression_model_code == 1:
			output_image = no_compression(input_image, output_image)												# create the output image without any compression
		# Shannon-Fano encoding
		if compression_model_code == 2:
			output_image = shannon_fano_decompression(input_image, input_key, symbol_dictionary, output_image)		# create the output image using the symbol dictionary
		# Dictionary/LZW encoding
		if compression_model_code == 3:
			output_image = dictionary_lzw_decompression(input_image, input_key, string_table, output_image)			# create the output image using the symbol dictionary
		# Arithmetic encoding
		else: #compression_model_code == 4
			output_image = arithmetic_decompression(input_image, input_key, arith_freq_list, output_image)			# create the output image code using Arithmetic encoding

		#####NEED TO FINISH#####
		# Display the image

	# Exit program
	if selection_code == 2:

main()