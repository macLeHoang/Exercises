def hamming_distance(img_left, img_right, maximum_disparity = 7, side = 'left'):
  '''
  This function calculates hamming distance between two input tensors
  :param img_left: left image
  :param right_img: right image
  :param maximum_disparity: maximum translation value
  :param size: determine which image is used as reference image

  :return hamming distance matrix of two input image 
  '''
  imgLeft = np.float32(img_left)
  imgRight = np.float32(img_right)
  distance = np.zeros(shape = (img_left.shape[0], img_right.shape[1], maximum_disparity))

  for d in (range(maximum_disparity)):
    if side == 'left':
      kernel = np.asarray([[1., 0., d], [0., 1., 0.]])
      shiftedRight = cv.warpAffine(imgRight, kernel, (img_right.shape[1], img_right.shape[0]))
      xorImg = np.uint64(img_left) ^ np.uint64(shiftedRight)
    elif side == 'right':
      kernel = np.asarray([[1., 0., -d], [0., 1., 0.]])
      shiftedLeft = cv.warpAffine(imgLeft, kernel, (img_left.shape[1], img_left.shape[0]))
      xorImg = np.uint64(shiftedLeft) ^ np.uint64(img_right)

    while np.any(xorImg != 0):
      diff = np.sum(xorImg & 1, axis = -1) if len(xorImg.shape) > 2 else xorImg & 1 # calculate number of bits different
      assert diff.shape == distance[..., d].shape
      distance[..., d] = distance[..., d] + diff
      xorImg = xorImg >> 1
  return distance

# Using 16 direction 
# dir_x = [0, -1, -1, -1, 0, 1, 1, 1, -1, -2, -2, -1, 1, 2, 2, 1]
# dir_y = [1, 1, 0, -1, -1, -1, 0, 1, 2, 1, -1, -2, -2, -1, 1, 2]
# num_dirs =  16

def semi_global_path_cost(img_left, img_right, maximum_disparity = 7, side = 'left', P1 = 1.0, P2 = 10.0):
  '''
    :param img_left: left image
    :param img_right: right image
    :param maximum_disparity: maximum translation value
    :param side: determine which image is used as reference
    :param P1: small penalty
    :param P2: large penalty

    :return semi_global_path_cost, with shape (imgH, imgW, maximum_disparity, 16)
  '''
  dir_x = [0, -1, -1, -1, 0, 1, 1, 1, -1, -2, -2, -1, 1, 2, 2, 1]
  dir_y = [1, 1, 0, -1, -1, -1, 0, 1, 2, 1, -1, -2, -2, -1, 1, 2]
  num_dirs =  16

  # print('Hamming distance calculation: ')
  hammingDistanceMat = hamming_distance(img_left, img_right, maximum_disparity, side)  # Cost Matrix
  offset = 2
  matH, matW, matD = hammingDistanceMat.shape # imgH, imgW, maximum_disparity

  # Padding Hamming Distance Matrix with INF value
  paddedHammingDistanceMat = np.full(shape = (matH + 4,
                                              matW + 4,
                                              matD + 2), fill_value = 128)
  paddedHammingDistanceMat[offset : offset + matH,
                           offset : offset + matW, 
                           1 : -1] = hammingDistanceMat
  path = np.zeros(shape = (img_left.shape[0],
                           img_right.shape[1],
                           maximum_disparity,
                           num_dirs), dtype = np.float64) # shape = imgH, imgW, maximum_disparity, 16
  # print('Semi-Global Matching: ')
  for idx, (dy, dx) in (enumerate(zip(dir_y, dir_x))):
    path[..., idx] = hammingDistanceMat
    # Min value of cost function in depth at offset (dy, dx)
    minCostMat = np.min(paddedHammingDistanceMat[offset + dy : offset + dy + matH,
                                                 offset + dx : offset + dx + matW, 
                                                 :], axis = -1) # shape = (matH, matW)

    # c1 - Cost of pixel in (dy, dx) direction at d-th position in depth of Hamming distance Matrix 
    c1 = paddedHammingDistanceMat[offset + dy : offset + dy + matH,
                                  offset + dx : offset + dx + matW, 
                                  1 : -1, 
                                  np.newaxis] # shape = (matH, matW, matD, 1)
    # c2 - Cost of pixel in (dy, dx) direction at (d - 1)-th position in depth of Hamming distance Matrix plus small penalty
    c2 = paddedHammingDistanceMat[offset + dy : offset + dy + matH,
                                  offset + dx : offset + dx + matW, 
                                  0 : -2,
                                  np.newaxis] + P1 # shape = (matH, matW, matD, 1)
    # c3 - Cost of pixel in (dy, dx) direction at (d + 1)-th position in depth of Hamming distance Matrix plus small penalty
    c3 = paddedHammingDistanceMat[offset + dy : offset + dy + matH,
                                  offset + dx : offset + dx + matW, 
                                  2 :,
                                  np.newaxis] + P1 # shape = (matH, matW, matD, 1)
    # c4 - Min value (in depth) of the Hamming distance matrix at pixel in (dy, dx) direction
    c4 = np.repeat(minCostMat[..., np.newaxis],
                   repeats = matD, axis = -1)[..., np.newaxis] + P2 # shape = (matH, matW, matD, 1)
    c5 = np.repeat(minCostMat[..., np.newaxis],
                   repeats = matD, axis = -1) # shape = (matH, matW, matD)               
    c_ = np.concatenate([c1, c2, c3, c4], axis = -1) # shape = (matH, matW, matD, 4)
    path[..., idx] = path[..., idx] + (np.min(c_, axis = -1) - c5)
  return path  
