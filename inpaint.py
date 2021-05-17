%matplotlib inline
from google.colab.patches import cv2_imshow
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image


root = "/content/drive/MyDrive/Segmentation/multifocus segment/mfs/ppt/inpaint/"
for im in os.listdir(root):
    src=cv2.imread(root+im)
    print( src.shape )
    #cv2.imshow("original Image" , src )


    # Convert the original image to grayscale
    grayScale = cv2.cvtColor( src, cv2.COLOR_RGB2GRAY )
    cv2_imshow(grayScale)
    #cv2.imwrite('grayScale_sample1.jpg', grayScale, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1,(17,17))
    
    # Perform the blackHat filtering on the grayscale image to find the 
    # hair countours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    cv2_imshow(blackhat)
    #cv2.imwrite('blackhat_sample1.jpg', blackhat, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # intensify the hair countours in preparation for the inpainting 
    # algorithm
    ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    print( thresh2.shape )
    cv2_imshow(thresh2)
    #cv2.imwrite('thresholded_sample1.jpg', thresh2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # inpaint the original image depending on the mask
    dst = cv2.inpaint(src,thresh2,1,cv2.INPAINT_TELEA)
    cv2_imshow(dst)
    cv2.imwrite("/content/drive/MyDrive/Segmentation Ram sir/multifocus segment/mfs/ppt/inpaint/result"+im+".png", dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
