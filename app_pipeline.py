# import numpy as np
import cv2
# import imutils
import os
# import matplotlib.pyplot as plt
# from image_processing import *
import image_processing
from server.scan import *

PATH = os.path.abspath(os.curdir) + r'\data\\'
img_name = 'img6.jpg'
img_path = PATH + img_name

if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_ = resize(img, new_height=1800, new_width=1800)
    proc_img1 = preprocess(img_)
    # fixed = image_processing.deskew(img_)
    # image_processing.display(fixed)
    #
    fixed_contour = get_contours(proc_img1, img_)
    warped_img = getWarp(img_, fixed_contour)

    display(warped_img)


# if __name__ == '__main__':
#     img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
#     img_ = image_processing.resize(img, new_height=1800, new_width=1800)
#     proc_img1 = image_processing.preprocess(img_)
#     # fixed = image_processing.deskew(img_)
#     # image_processing.display(fixed)
#     #
#     fixed_contour = image_processing.get_contours(proc_img1, img_)
#     warped_img = image_processing.getWarp(img_, fixed_contour)
#
#     image_processing.display(warped_img)
