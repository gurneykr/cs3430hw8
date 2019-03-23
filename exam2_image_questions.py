import math
import numpy as np
import argparse
import cv2
import sys
import os
import re
import statistics

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

def find_mean(arr):
    sum = 0
    count = 0
    for row in arr:
        for col in row:
            sum += col
            count += 1
    return sum/count

def find_median(arr):
    new_arr = []

    for row in arr:
        for col in row:
            new_arr.append(col)

    sort = sorted(new_arr)

    return sort[int(len(new_arr)/2)]

def find_mode(arr):
    new_arr = []

    for row in arr:
        for col in row:
            new_arr.append(col)

    return statistics.mode(new_arr)

def mean_median_mode(image_path):
    image = cv2.imread(image_path)
    B, G, R = cv2.split(image)

    blue_mean = find_mean(B)
    green_mean = find_mean(G)
    red_mean = find_mean(R)

    blue_median = find_median(B)
    green_median = find_median(G)
    red_median = find_median(R)

    blue_mode = find_mode(B)
    green_mode = find_mode(G)
    red_mode  = find_mode(R)

    return {"mean":(blue_mean, green_mean, red_mean), "median":(blue_median, green_median, red_median), "mode":(blue_mode, green_mode, red_mode)}

# def amplify_grayscale_blur_img_dir(ftype, in_img_dir, kz, c, amount):
#
#     imglst = read_img_dir(ftype, in_img_dir)
#     index = 0
#     for i in imglst:
#         im_amplified = amplify(index, imglst, c, amount)
#         im_gray = grayscale(index, imglst)
#         kernel = np.ones((kz,kz), np.float32)/(kz**2)
#         blurred = cv2.filter2D(im_gray, -1, kernel)
#         cv2.imwrite(imglst[index][0].split(".")[0]+"_blur.jpg", blurred)
#         index += 1

## here is main for you to test your implementations.
## remember to destroy all windows after you are done.
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # ap.add_argument('-t', '--type', required=True, help='Type of image')
    # ap.add_argument('-p', '--path', required=True, help='Path to image directory')
    # args = vars(ap.parse_args())

    # amplify_grayscale_blur_img_dir(args['type'], args['path'], 15, 'g', 100)
    print(mean_median_mode('C:\\Users\\Krista Gurney\\Documents\\cs3430\\Exam2Review\\images\\output11839.jpg'))
    cv2.waitKey()
    cv2.destroyAllWindows()