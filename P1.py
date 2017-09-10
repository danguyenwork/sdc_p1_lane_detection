#importing some useful packages

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import sys

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray,cmap='gray')"""

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon formed from
    `vertice`. The rest of the image is set to black
    """

    # return an array of zeros with the same shape and type as a given array
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def fit_lane_lines(lines):
    slopes = (lines[:,3] - lines[:,1]) / (lines[:,2] - lines[:,0])
    left_slope_index = slopes < 0
    left_slope = slopes[left_slope_index].mean()
    right_slope = slopes[~left_slope_index].mean()

    left_b = (lines[left_slope_index][:,1] - lines[left_slope_index][:,0] * left_slope).mean()
    right_b = (lines[~left_slope_index][:,3] - lines[~left_slope_index][:,2] * right_slope).mean()

    return left_slope, right_slope, left_b, right_b

def draw_lines(img, lines, Y, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    lines = lines[:,0]
    left_slope, right_slope, left_b, right_b = fit_lane_lines(lines)

    # import ipdb; ipdb.set_trace()
    top_y = lines[:,3].min()

    x_bottom_right = int((Y - right_b) / right_slope)
    x_bottom_left =  int((Y - left_b) / left_slope)

    x_top_right = int((top_y - right_b) / right_slope)
    x_top_left = int((top_y - left_b) / left_slope)

    cv2.line(img, (x_bottom_right, Y), (x_top_right, top_y), color, thickness)
    cv2.line(img, (x_bottom_left, Y), (x_top_left, top_y), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, Y):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, Y, thickness=5)
    return line_img

def weighted_img(img, initial_img, alpha=0.8, beta=1., lam=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * alpha + img * beta + lam
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, lam)

def process_image(image):
    copy = np.copy(image)
    copy = grayscale(copy)

    copy = gaussian_blur(copy, 5)

    copy = canny(copy, 20, 150)

    Y = copy.shape[0]
    X = copy.shape[1]
    BOTTOM_LEFT = (int(150 / 960. * X),Y)
    TOP_LEFT = (int(460 / 960. * X), int(320 / 540. * Y))
    TOP_RIGHT = (int(500 / 960. * X), int(320 / 540. * Y))
    BOTTOM_RIGHT = (int(860 / 960. * X),Y)

    vertices = np.array([[BOTTOM_LEFT, TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT]], dtype=np.int32)
    copy = region_of_interest(copy, vertices)

    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15    # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 100
    copy = hough_lines(copy, rho, theta, threshold, min_line_length, max_line_gap, Y)
    copy = weighted_img(copy,image,alpha=0.8, beta=1.2, lam=0.)

    return copy

def image_test(directory):
    arr = os.listdir(directory)
    for filename in arr:
        filename_arg = os.path.splitext(filename)
        if '_out' not in filename_arg[0] and filename_arg[1] in ['.jpg','.png']:
            inp = mpimg.imread(directory + filename)
            output = process_image(inp)
            output_dir = directory+filename_arg[0]+'_out'+filename_arg[1]
            mpimg.imsave(output_dir,output)

def video_test(directory):
    arr = os.listdir(directory)
    for filename in arr:
        filename_arg = os.path.splitext(filename)
        if '_out' not in filename_arg[0] and filename_arg[1] in ['.mp4']:
            print('processing input: ' + filename)
            inp = VideoFileClip(directory + filename).subclip(0,0)
            print('processing lane detection: ' + filename)
            output = inp.fl_image(process_image)
            print('processing output: ' + filename)
            output_dir = directory+filename_arg[0]+'_out'+filename_arg[1]
            output.write_videofile(output_dir, audio=False)

VID_DIR = "test_videos/"
IMG_DIR = "test_images/"


if __name__ == '__main__':
    if len(sys.argv) == 1:
        image_test(IMG_DIR)
        video_test(VID_DIR)
    elif sys.argv[1] == 'img':
        image_test(IMG_DIR)
    elif sys.argv[1] == 'vid':
        video_test(VID_DIR)
