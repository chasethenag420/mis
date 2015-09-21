import colorsys as cs
import numpy as np
from colormath.color_objects import XYZColor, sRGBColor, LabColor
from colormath.color_conversions import convert_color

# helper to call associated convertion function for given color mode and color instance
def get_color_values_in_rgb( first, second, third, color_model):
  if color_model == "RGB":
    return (first,second,third)
  elif color_model == "XYZ":
    return xyz2rgb(first,second,third)
  elif color_model == "Lab":
    return lab2rgb(first,second,third)
  elif color_model == "YUV":
    return yuv2rgb(first,second,third)
  elif color_model == "YCbCr":
    return ycbcr2rgb(first,second,third)
  elif color_model == "YIQ":
    return yiq2rgb(first,second,third)
  elif color_model == "HSL":
    return hsl2rgb(first,second,third)
  elif color_model == "HSV":
    return hsv2rgb(first,second,third)
  else :
    return (first,second,third)

# converts xyz color instace to rgb
# input is unnormalized values
def xyz2rgb(x,y,z):
  "XYZ: x [0,95.047]  y[0,100.000]  z[0,108.883]"
  xyz = XYZColor(x/100.0, y/100.0, z/100.0,observer='2',illuminant='d65')
  rgb = convert_color(xyz, sRGBColor)
  return rgb.get_upscaled_value_tuple()

# converts rgb color instace to xyz
# input is unnormalized values
def rgb2xyz(r,g,b):
  "RGB: R[0, 255] g[0, 255] b[0, 255]"
  rgb = sRGBColor(r/255.0, g/255.0, b/255.0)
  xyz = convert_color(rgb, XYZColor,target_illuminant='d65')
  return tuple(int(round(i*100)) for i in xyz.get_value_tuple())

# converts lab color instace to rgb
# input is unnormalized values
def lab2rgb(l,a,b):
  "LAB: l[0, 100] a[-86.185, 98,254] b[-107.863, 94.482]"
  lab = LabColor(l, a, b)
  rgb = convert_color(lab, sRGBColor)
  return rgb.get_upscaled_value_tuple()

# converts rgb color instace to lab
# input is unnormalized values
def rgb2lab(r,g,b):
  "RGB: R[0, 255] g[0, 255] b[0, 255]"
  rgb = sRGBColor(r/255.0, g/255.0, b/255.0)
  lab = convert_color(rgb, LabColor)
  return tuple(int(round(i)) for i in lab.get_value_tuple())

# converts yuv color instace to rgb
# input is unnormalized values
def yuv2rgb(y,u,v):
  "YUV: Y [0.0, 255] - U [-127, 128] - V [-127, 128]"
  yuv=np.array([(y/255.0,),(u/255.0,),(v/255.0,)])
  tranformation=np.array([(1.000,0.000,1.140),(1.000,-0.395,-0.581),(1.000,2.032,0.000)])
  rgb=np.dot(tranformation, yuv)
  return tuple(int(round(i*255)) for i in rgb)

# converts rgb color instace to yuv
# input is unnormalized values
def rgb2yuv(r,g,b):
  "RGB: R[0, 255] g[0, 255] b[0, 255]"
  rgb=np.array([(r/255.0,),(g/255.0,),(b/255.0,)])
  tranformation=np.array([(0.299,0.587,0.114),(-0.147,-0.289,0.436),(0.615,-0.515,-0.100)])
  yuv=np.dot(tranformation, rgb)
  return tuple(int(round(i*255)) for i in yuv)

# converts rgb color instace to ycbcr
# input is unnormalized values
def rgb2ycbcr(r,g,b) :
  "RGB: R[0, 255] g[0, 255] b[0, 255]"
  rgb =np.array([(r,),(g,),(b,)])
  tranformation=np.array([(0.299,0.587,0.114),(-0.169,-0.331,0.500),(0.500,-0.419,-0.081)])
  addition = np.array([(0,),(128,),(128,)])
  ycbcr= np.dot(tranformation,rgb) + addition
  return tuple(int(round(i)) for i in ycbcr)

# converts yccbcr color instace to rgb
# input is unnormalized values
def ycbcr2rgb(y,cb,cr):
  "YCbCr: Y [0, 255] - Cb [0, 255] - Cr [0, 255]"
  ycbcr=np.array([(y,),(cb-128,),(cr-128,)])
  tranformation=np.array([(1.0,0.0,1.4),(1.0,-0.343,-0.711),(1.0,1.765,0.000)])
  rgb=np.dot(tranformation, ycbcr)
  return tuple(int(round(i)) for i in rgb)

# converts rgb color instace to yiq
# input is unnormalized values
def rgb2yiq(r,g,b):
  "RGB: R[0, 255] g[0, 255] b[0, 255]"
  r = r/255.0
  g = g/255.0
  b = b/255.0
  return tuple(int(round(i * 255)) for i in cs.rgb_to_yiq(r,g,b) )

# converts yiq color instace to rgb
# input is unnormalized values
def yiq2rgb(y,i,q):
  "YIQ: Y [0.0, 255] - I [-127, 128] - Q [-127, 128]"
  y = y/255.0
  i = i/255.0
  q = q/255.0
  return tuple(int(round(i * 255)) for i in cs.yiq_to_rgb(y,i,q) )

# converts hsl color instace to rgb
# input is unnormalized values
def hsl2rgb(h,s,l):
  "HSL: H [0, 360] - L [0.0, 100] - S [0.0, 100]"
  h = h/360.0
  s = s/100.0
  l = l/100.0
  return tuple(int(round(i * 255)) for i in cs.hls_to_rgb(h,l,s) )

# converts rgb color instace to hsl
# input is unnormalized values
def rgb2hsl(r,g,b):
  "RGB: R[0, 255] g[0, 255] b[0, 255]"
  r = r/255.0
  g = g/255.0
  b = b/255.0
  hls = cs.rgb_to_hls(r,g,b)
  return (int(round(hls[0]*360)),int(round(hls[2]*100)),int(round(hls[1]*100)))

# converts hsv color instace to rgb
# input is unnormalized values
def hsv2rgb(h,s,v) :
  "HSV: H [0, 360] - S [0.0, 100] - V [0.0, 100]"
  h = h/360.0
  s = s/100.0
  v = v/100.0
  return tuple(int(round(i * 255)) for i in cs.hsv_to_rgb(h,s,v))

# converts rgb color instace to hsv
# input is unnormalized values
def rgb2hsv(r,g,b):
  "RGB: R[0, 255] g[0, 255] b[0, 255]"
  r = r/255.0
  g = g/255.0
  b = b/255.0
  hsv = cs.rgb_to_hsv(r,g,b)
  return (int(round(hsv[0]*360)),int(round(hsv[1]*100)),int(round(hsv[2]*100)))

# test values
def test():
  print 'xyz2rgb ' + str(xyz2rgb(48,37,5))
  print 'rgb2xyz ' + str(rgb2xyz(250,132,14))

  print 'lab2rgb ' + str(lab2rgb(67,39,72))
  print 'rgb2lab ' + str(rgb2lab(250,132,14))

  print 'yiq2rgb ' + str(yiq2rgb(154,108,-11))
  print 'rgb2yiq ' + str(rgb2yiq(250,132,14))

  print 'yuv2rgb ' + str(yuv2rgb(154, -69, 84))
  print 'rgb2yuv ' + str(rgb2yuv(250,132,14))

  print 'ycbcr2rgb ' + str(ycbcr2rgb(154,49,197))
  print 'rgb2ycbcr ' + str(rgb2ycbcr(250,132,14))

  print 'hsv2rgb ' + str(hsv2rgb(30,94,98))
  print 'rgb2hsv ' + str(rgb2hsv(250,132,14))

  print 'hsl2rgb ' + str(hsl2rgb(30,96,52))
  print 'rgb2hsl ' + str(rgb2hsl(250,132,14))
