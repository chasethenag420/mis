import cv2
import sys
import numpy as np
import os

def main():
  option_number=raw_input("Enter the option number:\n")
  input_file_name=raw_input("Enter the input file name of tpc or spc:\n")  
  input_file_name_split=input_file_name.split('.')
  suffix=None
  
  if input_file_name_split[1]=="tpc":
    suffix=option_number+".tpq"
  else:
    suffix=option_number+".spq"

  output_file_name=input_file_name_split[0]+"_"+suffix

  current_working_dir = os.getcwd()
  infile = open( input_file_name ) 
  outfile = open( output_file_name,'w' ) 
  
  frames= None 

  if input_file_name != None:
    if option_number=='1':
      frames=quantizationOption1(infile,outfile) 
    elif option_number=='2':
      frames=quantizationOption2(infile,outfile)
    else: 
      print "Input not valid"
      quit()
  else:
    print "Some error while reading input file"

  infile.flush()
  infile.close()

  outfile.flush()
  outfile.close()

  
def quantizationOption1(infile,outfile):
   
  for line in infile:
    outfile.write(line)
    
def quantizationOption2(infile,outfile):
   
  numOfBits=int(raw_input("Enter number of bits m: "))
  numOfPartitions = 2 ** numOfBits
  for line in infile:
    signal= list(map(float,line.split()))
    #signal=[1,2,3,4,5,6,7,8,9]
    maxValue = max(signal)
    minValue = min(signal)
    partitionSize = (maxValue - minValue)/float(numOfPartitions)
    partitions=None
    if partitionSize != 0 :
      partitions = np.arange(minValue, maxValue+partitionSize, partitionSize)
    else :
      partitions = np.ones(numOfPartitions+1) * minValue
    
    binIndexes=np.digitize(np.array(signal),partitions)
    representative=[]
    for value in range(1,len(partitions)):
      representative.append(partitions[value-1]/float(2)+partitions[value]/float(2))

    quantized=[]

    for i in binIndexes.tolist():
      index=0
      if i-1 >= len(representative):
        index= i-2
      else:
        index=i-1
      quantized.append(representative[index])
    outfile.write(" ".join(map(str,quantized))+"\n")

main()
