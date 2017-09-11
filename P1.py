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

def to_hsl(img):
    """Applies the HSL transform"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray,cmap='gray')"""

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def isolate_colors(img):
    """Isolate yellow and white to improve recognition
    img should be in OpenCV HSL format
    """

    boundaries = [([15, 38, 115], [35, 204, 255]),([0, 200, 0], [180, 255, 255])]
    outputs = []

    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask = mask)
        outputs.append(output)

    return sum(outputs)

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


def extract_potential_lanes(lines):
    """
    Helper for draw_lines
    Eliminate lines where the slopes are above or below a certain threshold.
    This gets rid of things like shadows, smudges, etc.
    """
    slopes = (lines[:,3] - lines[:,1]) / (lines[:,2] - lines[:,0])
    valid_line_index = (((slopes < .9) & (slopes > .4)) | ((slopes > -.9) & (slopes < -.4)))
    valid_lines = lines[valid_line_index]
    return(valid_lines)

def parse_lane_points(lines):
    """
    Helper for draw_lines
    Prepare data for polyfit
    Parse the list of lines into four sets of arrays: x and y coordinates of left and right lanes
    """
    slopes = (lines[:,3] - lines[:,1]) / (lines[:,2] - lines[:,0])
    left_lane_index = slopes < 0
    left_lane_points_x = lines[left_lane_index][:,[0,2]].reshape(1,-1)[0]
    left_lane_points_y = lines[left_lane_index][:,[1,3]].reshape(1,-1)[0]
    right_lane_points_x = lines[~left_lane_index][:,[0,2]].reshape(1,-1)[0]
    right_lane_points_y = lines[~left_lane_index][:,[1,3]].reshape(1,-1)[0]
    return left_lane_points_x, left_lane_points_y, right_lane_points_x, right_lane_points_y

def draw_hough_line(img, lines, color=[255,0,0], thickness=2):
    """
    Draw hough lines without any extrapolation
    """
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img, (x1,y1), (x2,y2),[255, 0, 0],2)
    return line_img

def draw_lines(img, lines, Y, color=[255, 0, 0], thickness=2):
    """
    This function extrapolates from a set of lines to arrive at the final lanes.
    - Eliminate invalid lines
    - Parse points into left and right lane based on slopes
    - Polyfit the data to calculate the slope and intercept of each lane
    - Calculate the top of the region of interest
    - Calculate the end-points of each lane
    - Draw each lane on a copy of the image
    Return a copy of the image with the two lanes drawn
    """

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    valid_lines = extract_potential_lanes(lines[:,0])

    left_lane_points_x, left_lane_points_y, right_lane_points_x, right_lane_points_y = parse_lane_points(valid_lines)

    top_y = max(len(left_lane_points_y) != 0 and left_lane_points_y.min() or 0, \
                len(right_lane_points_y) != 0 and right_lane_points_y.min() or 0)

    if len(left_lane_points_x) != 0:
        left_lane = np.polyfit(left_lane_points_x, left_lane_points_y, 1)
        x_bottom_left =  int((Y - left_lane[1]) / left_lane[0])
        x_top_left = int((top_y - left_lane[1]) / left_lane[0])
        cv2.line(line_img, (x_bottom_left, Y), (x_top_left, top_y), color, thickness)

    if len(right_lane_points_y) != 0:
        right_lane = np.polyfit(right_lane_points_x, right_lane_points_y, 1)
        x_bottom_right = int((Y - right_lane[1]) / right_lane[0])
        x_top_right = int((top_y - right_lane[1]) / right_lane[0])
        cv2.line(line_img, (x_bottom_right, Y), (x_top_right, top_y), color, thickness)
    return line_img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

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


def process_image(image, filename="default"):
    """ pipeline for lane detection:
    - HSL
    - Isolate yellow / white
    - Grayscale
    - Gaussian
    - Canny
    - Hough
    - Lane extrapolation
    """

    copy = np.copy(image)

    # ==================
    # Configurable parameters
    Y = copy.shape[0]
    X = copy.shape[1]
    BOTTOM_LEFT = (int(150 / 960. * X),Y)
    TOP_LEFT = (int(450 / 960. * X), int(310 / 540. * Y))
    TOP_RIGHT = (int(500 / 960. * X), int(310 / 540. * Y))
    BOTTOM_RIGHT = (int(920 / 960. * X),Y)
    vertices = np.array([[BOTTOM_LEFT, TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT]], dtype=np.int32)

    rho = 1   # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15    # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 40
    # ==================

    copy = to_hsl(copy)
    filename != 'default' and mpimg.imsave('test_images/' + filename + '_out_0hsl.png',copy)

    copy = isolate_colors(copy)
    filename != 'default' and mpimg.imsave('test_images/' + filename + '_out_1isolate.png',copy)

    copy = grayscale(copy)
    filename != 'default' and mpimg.imsave('test_images/' + filename + '_out_2grayscale.png',copy,cmap='gray')

    copy = gaussian_blur(copy, 5)
    filename != 'default' and mpimg.imsave('test_images/' + filename + '_out_3gaussian.png',copy,cmap='gray')

    copy = canny(copy, 50, 150)
    filename != 'default' and mpimg.imsave('test_images/' + filename + '_out_4canny.png',copy,cmap='gray')

    copy = region_of_interest(copy, vertices)
    filename != 'default' and mpimg.imsave('test_images/' + filename + '_out_5region.png',copy,cmap='gray')

    lines = hough_lines(copy, rho, theta, threshold, min_line_length, max_line_gap)
    copy2 = draw_hough_line(copy, lines)
    filename != 'default' and mpimg.imsave('test_images/' + filename + '_out_6hough.png',copy2,cmap='gray')

    copy = draw_lines(copy, lines, Y, thickness=5)
    filename != 'default' and mpimg.imsave('test_images/' + filename + '_out_7drawline.png',copy,cmap='gray')

    copy = weighted_img(copy,image,alpha=0.8, beta=1.2, lam=0.)
    return copy

def process_directory(directory):
    arr = os.listdir(directory)
    for filename in arr:
        print('processing input: ' + filename)
        filename_arg = os.path.splitext(filename)
        output_dir = directory+filename_arg[0]+'_out_8final'+filename_arg[1]

        if '_out' in filename_arg[0]:
            continue
        elif filename_arg[1] in ['.jpg','.png']:
            inp = mpimg.imread(directory + filename)
            output = process_image(inp, filename_arg[0])
            mpimg.imsave(output_dir,output)
        elif filename_arg[1] in ['.mp4']:
            inp = VideoFileClip(directory + filename)
            output = inp.fl_image(process_image)
            output.write_videofile(output_dir, audio=False)

VID_DIR = "test_videos/"
IMG_DIR = "test_images/"

if __name__ == '__main__':
    # python P1.py
    if len(sys.argv) == 1:
        process_directory(IMG_DIR)
        process_directory(VID_DIR)
    # python P1.py img
    elif sys.argv[1] == 'img':
        process_directory(IMG_DIR)
    # python P1.py vid
    elif sys.argv[1] == 'vid':
        process_directory(VID_DIR)
