#!/usr/bin/python

#########################################
# module: hw08_s19.py
# Krista Gurney
# A01671888
#########################################

### modify these as you see fit.
import math
import numpy as np
import argparse
import cv2
import sys
import os
import re

def generate_file_names(ftype, rootdir):
    '''
    recursively walk dir tree beginning from rootdir
    and generate full paths to all files that end with ftype.
    sample call: generate_file_names('.jpg', /home/pi/images/')
    '''
    for path, dirlist, filelist in os.walk(rootdir):
        for file_name in filelist:
            if not file_name.startswith('.') and \
               file_name.endswith(ftype):
                yield os.path.join(path, file_name)
        for d in dirlist:
            generate_file_names(ftype, d)

def read_img_dir(ftype, imgdir):
    images_array = []
    for file in generate_file_names(ftype,imgdir):
        images_array.append((file, cv2.imread(file)))
    return images_array

def luminosity(rgb, rcoeff=0.2126, gcoeff=0.7152, bcoeff=0.0722):
    return rcoeff*rgb[0]+gcoeff*rgb[1]+bcoeff*rgb[2]

def grayscale(i, imglst):

    for row in (imglst[i][1]):
        for col in row:
            lum = luminosity(col)
            col[0] = col[1] = col[2] = lum

    return imglst[i][1]


def amplify(i, imglst, c, amount):
    ## split the image into 3 channels
    B, G, R = cv2.split(imglst[i][1])

    if c == "b":
        amplified_blue = cv2.merge([B + amount, G, R])
        return amplified_blue
    elif c == "g":
        amplified_green = cv2.merge([B, G + amount, R])
        return amplified_green
    elif c == "r":
        amplified_red = cv2.merge([B, G, R + amount])
        return amplified_red

def amplify_grayscale_blur_img_dir(ftype, in_img_dir, kz, c, amount):

    imglst = read_img_dir(ftype, in_img_dir)
    index = 0
    for i in imglst:
        im_amplified = amplify(index, imglst, c, amount)
        im_gray = grayscale(index, imglst)
        kernel = np.ones((kz,kz), np.float32)/(kz**2)
        blurred = cv2.filter2D(im_gray, -1, kernel)
        cv2.imwrite(imglst[index][0].split(".")[0]+"_blur.jpg", blurred)
        index += 1

## here is main for you to test your implementations.
## remember to destroy all windows after you are done.
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--type', required=True, help='Type of image')
    ap.add_argument('-p', '--path', required=True, help='Path to image directory')
    args = vars(ap.parse_args())

    amplify_grayscale_blur_img_dir(args['type'], args['path'], 15, 'g', 100)

    cv2.waitKey()
    cv2.destroyAllWindows()


    


 
