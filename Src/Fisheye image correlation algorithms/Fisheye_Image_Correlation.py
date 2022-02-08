import cv2 as cv
import numpy as np
import math

def get_basic_information(img):
  #radius of fisheye image
  height = img.shape[0]
  width = img.shape[1]
  R = height // 2 if height < width else width // 2

  #coordinates of origin
  x_0 = width // 2
  y_0 = height // 2
  return [height, width, R, x_0, y_0]
  
  
def interpolation(origin_image, x, y):
  x = int(x)
  y = int(y)
  pixel = origin_image[y, x, :]
  return pixel
  
def spherical_coordinates_positioning_correction(img, height, width, R, x_0, y_0):
  #intial corrected image whose shape is the same with origin image's shape
  c_img = np.zeros(shape = (height + 1, width + 1, 3))
  
  for u in range(width):
    for v in range(height):
      x = x_0 + (u - x_0) * (R**2 - (v - y_0)**2)**0.5 / R if R > (v - y_0) else x_0 
      y = v
      c_img[v, u, :] = interpolation(img, x, y)
      
  c_img = np.asarray(c_img, dtype = np.uint8)
  return c_img[:height, :width, :]
  
  
def spherical_perspective_projection(img, height, width, R, x_0, y_0):
  c_img = np.zeros(shape = img.shape)
  
  for v in range(height):
    for u in range(width):
      x = x_0 + R * (u - x_0) / (R**2 + (u - x_0)**2 + (v - y_0)**2)**0.5
      y = y_0 + R * (v - y_0) / (R**2 + (u - x_0)**2 + (v - y_0)**2)**0.5

      c_img[v, u, :] = interpolation(img, x, y)      
  c_img = np.asarray(c_img, dtype = np.uint8)
  return c_img
  

def panoramic_expansion_model(img, height, width, R, x_0, y_0):
  c_height = R
  c_width = round(math.pi * R)
  c_img = np.zeros(shape = (c_height, c_width, 3))

  for v in range(c_height):
    for u in range(c_width):
      x = x_0 + v * math.cos(2*u / R)
      y = y_0 + v * math.sin(2*u / R)

      c_img[v, u, :] = interpolation(img, x, y)
  c_img = np.asarray(c_img, dtype = np.uint8)
  return c_img
  
def convert_video(path, cvtType = 'shperical_perspective_projection'):
  video = cv.VideoCapture(path)
  basic_list = list()
  count = 1

  while (vid.isOpen()):
    _, frame = video.read()

    if count == 1:
      basic_list = get_basic_information(frame)
      count == 0

    #cvtFrame = shperical_perspective_projection(frame, *basic_list)
    cv.imshow(cvtFrame)

    if cv.waitKey(1) == ord('q'):
      break
  
  video.release()
