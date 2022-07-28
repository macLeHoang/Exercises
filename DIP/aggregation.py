import numpy as np

from LBPs import LBP_transform, SLBP_transform, LTP_transform, LDP_transform, LTrP_transform
from semi_global import hamming_distance, semi_global_path_cost

def aggregation_path_cost(img_left, img_right, maximum_disparity = 7, side = 'left', P1 = 1.0, P2 = 10.0, normalize = True, ksize = None):
  '''
  Sum of all direction cost of SGM at pixel (h, w)
    :return position of the minimum aggregation cost (in depth) at pixel (h, w) 
  '''
   # shape = (imgH, imgW, maximum_disparity, the number of direction used for calculating path cost)
  SGM_path = semi_global_path_cost(img_left, img_right, maximum_disparity, side, P1, P2)
  aggregationCost = np.sum(SGM_path, axis = -1) # shape = (imgH, imgW, maximun_disparity)
  disparity = np.argmin(aggregationCost, axis = -1) # shape = (imgH, imgW)

  if normalize:
    ksize = 9 if ksize is None else ksize
    disparity = cv.medianBlur(np.uint8(disparity), ksize)
    disparity = (disparity - np.min(disparity)) / (np.max(disparity) - np.min(disparity))
  return disparity
