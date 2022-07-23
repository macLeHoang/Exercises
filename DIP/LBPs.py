import math
import numpy as np

def LBP_transform(img, npoints = 8, radius = 1.0, clock_wise = False, start_angle = 0.0, interpolation = 'bilinear'):
  '''
  Use to calculate local binary pattern of input image
    :param img: input image
    :param npoints: number of adjacent points used in calculation 
    :param radius: radius of circle 
    :param clock_wise: determine the direction of rotation
        if True: rotate clockwise - negative angle
        else: rotate counter-clockwise - positive angle
    :param start_angle: determine position of starting point 
    :param interpolation: should be 'nearest' or 'bilinear'
    
    :return LBP transform matrix 
  '''
  assert isinstance(npoints, int)
  assert interpolation in ('nearest', 'bilinear')

  imgH, imgW = img.shape[:2]
  lbpMatrix = np.zeros(shape = (imgH, imgW), dtype = np.uint)
  refX, refY = math.ceil(radius), math.ceil(radius)

  # Zeros padding input image
  padded = np.zeros(shape = (imgH + 2*math.ceil(radius), imgW + 2*math.ceil(radius)), dtype = np.uint)
  padded[math.ceil(radius):imgH + math.ceil(radius),
         math.ceil(radius):imgW + math.ceil(radius)] = img

  # Set up
  angle = 2 * math.pi / npoints # divide a circle into npoints-part equally
  rotateDirection = -1 if clock_wise else 1
  startAngle = start_angle / 180. * math.pi

  offset = []
  for i in range(npoints):
    offsetY = -radius * math.sin(startAngle + rotateDirection * angle * i)
    offsetX = radius * math.cos(startAngle + rotateDirection * angle * i)

    if interpolation == 'nearest':
      x = round(offsetX)
      y = round(offsetY)
      offset.append((y, x))

    elif interpolation == 'bilinear':
      offset.append((offsetY, offsetX))

  offset.reverse()
  assert len(offset) == npoints
  
  # nearest neightbor interpolation
  if interpolation == 'nearest':
    for (offsetY, offsetX) in offset:
      lbpMatrix = (lbpMatrix << 1) | (padded[refY + offsetY : refY + offsetY + imgH,
                                      refX + offsetX : refX + offsetX + imgW] >= img)
      
  # bilinear interpolation
  elif interpolation == 'bilinear':
    for (offsetY, offsetX) in offset:
      xmax, xmin = math.ceil(offsetX), math.floor(offsetX)
      ymax, ymin = math.ceil(offsetY), math.floor(offsetY)
      xWeight = offsetX - xmin
      yWeight = offsetY - ymin
      
      topLeft = padded[refY + ymin : refY + ymin + imgH,
                       refX + xmin : refX + xmin + imgW]
      topRight = padded[refY + ymin : refY + ymin + imgH,
                        refX + xmax: refX + xmax + imgW]
      bottomLeft = padded[refY + ymax : refY + ymax + imgH,
                          refX + xmin : refX + xmin + imgW]
      bottomRight = padded[refY + ymax : refY + ymax + imgH,
                           refX + xmax : refX + xmax + imgW]

      bilinearTop = (1 - xWeight) * topLeft + xWeight * topRight
      bilinearBottom = (1 - xWeight) * bottomLeft + xWeight * bottomRight

      bilinear = (1 - yWeight) * bilinearTop + yWeight * bilinearBottom

      lbpMatrix = (lbpMatrix << 1) | (bilinear >= img)
  return lbpMatrix


def SLBP_transform(img, npoints = 8, r = 1, interpolation = 'nearest'):
  '''
    :param img: input image
    :param npoints: number of adjacent neighbors, N <= 10
    :param r: radius 
    :param interpolation: must be 'nearest' or 'bilinear'

    :return SLBP transform matrix
  '''
  
  assert npoints <= 10 # Maximum number of bits used to store uint number of numpy is 64 bit

  offset = math.ceil(r)
  imgH, imgW = img.shape[:2]
  paddedImg = np.zeros(shape = (imgH + 2*offset, imgW + 2*offset))
  paddedImg[offset : offset + imgH, offset : offset + imgW] = img

  angleRadian = 2 * math.pi / npoints # divide circle into n-points part equally
  offsetPoints = []
  for i in range(1, npoints + 1):
    x = r * math.cos(i * angleRadian)
    y = -r * math.sin(i * angleRadian)
    offsetPoints.append((y, x)) # (angles of 45, 90, 135, ..., 360)
  
  slbpMatrix = np.zeros(shape = (imgH, imgW), dtype = np.uint64)

  if interpolation == 'nearest':
    for idx in range(npoints):
      if idx == 0:
        u_, v_ = (0, 0)
      else:
        u, v = offsetPoints[idx - 1]
        u_, v_ = round(u), round(v)
      subRegion = paddedImg[offset + u_ : offset + u_ + imgH, offset + v_ : offset + v_ + imgW]

      for (uu, vv) in offsetPoints[idx:]:
        uu_, vv_ = round(uu), round(vv)
        slbpMatrix = slbpMatrix << 1 | (subRegion <= paddedImg[offset + uu_ : offset + uu_ + imgH,
                                                               offset + vv_ : offset + vv_ + imgW])
  elif interpolation == 'bilinear':
    bilinear = np.zeros(shape = (imgH, imgW, npoints))
    for idx, (u, v) in enumerate(offsetPoints):
      umax, vmax = math.ceil(u), math.ceil(v)
      umin, vmin = math.floor(u), math.floor(v)
      weightU = u - umin
      weightV = v - vmin

      tl = (1 - weightV) * paddedImg[offset + umin : offset + umin + imgH,
                                      offset + vmin : offset + vmin + imgW]

      tr = weightV * paddedImg[offset + umin : offset + umin + imgH,
                               offset + vmax : offset + vmax + imgW] 

      bl = (1 - weightV) * paddedImg[offset + umax : offset + umax + imgH,
                                     offset + vmin : offset + vmin + imgW]

      br = weightV * paddedImg[offset + umax : offset + umax + imgH,
                               offset + vmax : offset + vmax + imgW]  
      
      bilinear[..., idx] = (1 - weightU)*(tl + tr) + weightU*(bl + br) # Assign bilinear value correponding to direction
                                                                       # e.g. idx = [0, 1, 2, 3, 4, 5, 6, 7] respectively corresponding to 
                                                                       # direction [45, 90, 135, 180, 225, 270, 315, 360]
    for idx in range(npoints):
      for idx2 in range(idx, npoints):
        if idx == 0:
          slbpMatrix = slbpMatrix << 1 | (img <= bilinear[..., idx2]) # LBP with Zc if direction is 0
        else:
          slbpMatrix = slbpMatrix << 1 | (bilinear[..., idx - 1] <= bilinear[..., idx2]) # Else swapping each value then doing LBP
  return slbpMatrix


def LTP_transform(img, window_size = 3, threshold = 1.0):
  '''
  Use to calculate local ternary pattern of input image
    :param img: input image
    :param window_size: use to consider neighborhood
    :param threshold: float

    :return LTP transform matrix 
  '''

  imgH, imgW = img.shape[:2]
  offset = window_size // 2
  
  paddedImg = np.zeros(shape = (imgH + offset * 2, imgW + offset * 2))
  paddedImg[offset : offset + imgH, 
            offset : offset + imgW] = img
  
  offsetList = [(u, v) for u in range(-offset, offset + 1) \
                       for v in range(-offset, offset + 1) \
                       if not (u == 0 and v == 0)]
  
  ltpPositive = np.zeros(shape = (imgH, imgW), dtype = np.uint)
  ltpNegative = np.zeros(shape = (imgH, imgW), dtype = np.uint)

  for (dy, dx) in offsetList:
    mask = paddedImg[offset + dy : offset + dy + imgH,
                     offset + dx : offset + dx + imgW]
    ltpPositive = (ltpPositive << 1) | (mask >= (img + threshold))
    ltpNegative = (ltpNegative << 1) | (mask <= (img - threshold))
  
  ltpMatrix = np.concatenate([ltpPositive[..., np.newaxis], ltpNegative[..., np.newaxis]], axis = -1)
  return ltpMatrix

def LDP_transform (img):
  '''
  Use in calculate local derivative pattern of input image
    :param img: input image

    :return LDP transform matrix
  '''
  translateList = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]  # respectively correspoding to angle of 0, 45, 90, 135 
  offset = 1

  imgH, imgW = img.shape[:2]
  paddedImg = np.zeros(shape = (imgH + 1, imgW + 2))
  paddedImg[1:, 1:-1] = img

  # offsetList = [(u, v) for u in range(-1, 2) for v in range(-1, 2) if not(u == 0 and v == 0)]
  offsetList = [(1, 1), (1, 0), (1, -1), (0, -1)]
  ldpMatrix = np.zeros(shape = (imgH, imgW), dtype = np.uint)

  for (dy, dx) in translateList:
    firstDerivative = img - paddedImg[offset + dy : offset + dy + imgH,
                                      offset + dx : offset + dx + imgW]

    paddedFirstDerivative = np.zeros(shape = (imgH + 2, imgW + 2))
    paddedFirstDerivative[1 : -1, 1 : -1] = firstDerivative

    for (u, v) in offsetList:
      mask = firstDerivative * paddedFirstDerivative[offset + u : offset + u + imgH,
                                                     offset + v : offset + v + imgW]
      ldpMatrix = (ldpMatrix << 1) | (mask <= 0)
  return ldpMatrix

def LTrP_transform(img, window_size = 3):
  offset = window_size // 2
  imgH, imgW = img.shape[:2]

  paddedImg = np.zeros(shape = (imgH + offset, imgW + offset))
  paddedImg[offset : offset + imgH,
             : imgW] = img
  
  dX = paddedImg[offset : offset + imgH,
                 offset : offset + imgW] - img # Horizontal First Derivative
  dY = paddedImg[:imgH, :imgW] - img # Vertical First Derivative

  mask1 = (dX >= 0) & (dY >= 0) 
  mask2 = (dX < 0)  & (dY >= 0)
  mask3 = (dX < 0)  & (dY < 0)
  mask4 = (dX >= 0) & (dY < 0)

  directionOfCenter = np.full(shape = (imgH, imgW), fill_value = 1) * mask1 +\
                      np.full(shape = (imgH, imgW), fill_value = 2) * mask2 +\
                      np.full(shape = (imgH, imgW), fill_value = 3) * mask3 +\
                      np.full(shape = (imgH, imgW), fill_value = 4) * mask4
                      
  paddedDirectionOfCenter = np.zeros(shape = (imgH + 2*offset, imgW + 2*offset))
  paddedDirectionOfCenter[offset : offset + imgH, 
                          offset : offset + imgW] = directionOfCenter
  
  # magnitude = dX**2 + dY**2
  # paddedMagnitude = np.zeros(shape = (imgH + 2, imgW + 2))
  # paddedMagnitude[1:-1, 1:-1] = magnitude

  # offsetList = [(u, v) for u in range(-1, 2) for v in range(-1, 2) if not(u == 0 and v == 0)]
  offsetList = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]

  # magnitudeEncodeMatrix = np.zeros(shape = (imgH, imgW), dtype = np.uint64)
  directionEncodeMatrix = np.zeros(shape = (imgH, imgW, 4), dtype = np.uint64)

  for (u, v) in offsetList:
    # magnitudeEncodeMatrix = magnitudeEncodeMatrix << 1 | (paddedMagnitude[1 + u : 1 + u + imgH,
    #                                                                       1 + v : 1 + v + imgW] >= magnitude)
    
    subRegion = paddedDirectionOfCenter[offset + u : offset + u + imgH,
                                        offset + v : offset + v + imgW] # neighbor in specific direction of centers

    mask_ = directionOfCenter != subRegion # Compare direction of center and its corresponding neighbor
                                           # True at position where the direction of center is different the direction of neighbors
                                        
    
    # Encode four directions
    directionEncodeMatrix[..., 0] = directionEncodeMatrix[..., 0] << 1 | (mask_ * subRegion == 1)
    directionEncodeMatrix[..., 1] = directionEncodeMatrix[..., 1] << 1 | (mask_ * subRegion == 2)
    directionEncodeMatrix[..., 2] = directionEncodeMatrix[..., 2] << 1 | (mask_ * subRegion == 3)
    directionEncodeMatrix[..., 3] = directionEncodeMatrix[..., 3] << 1 | (mask_ * subRegion == 4)
  return directionEncodeMatrix
