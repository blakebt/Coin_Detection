"""
Brenden Blake
CS 456
Project

Using image processing techniques to help in the detection of American coins
"""
import numpy as np
import cv2

# threshhold method, accepts an img and a lower bound
def threshold(img, lower):
    [row,col] = np.shape(img)
    threshold_img = np.zeros((row,col))
    threshold_img = np.uint8(threshold_img)

    for i in range(row):
        for j in range(col):
            if gray_img[i][j] > lower:
                threshold_img[i][j] = 0
            else:
                threshold_img[i][j] = 255

    return threshold_img

# method used in convolve method
def sum_of_products(img, x, y, w):
    m,n = np.shape(w)
    mid_m = m // 2
    mid_n = n // 2
    result = 0
    w_i = 0

    for i in range(x-mid_m, x+mid_m-1):
        w_j = 0
        for j in range(y-mid_n, y+mid_n-1):
            result += img[i][j] * w[w_i][w_j]
            w_j += 1
        w_i += 1

    return result
    
# convolution method
def convolve(img, w):
    row,col = np.shape(img)
    m,n = np.shape(w)

    w = np.rot90(w, 2)
    mid_m = m // 2
    mid_n = n // 2
    temp_img = np.zeros((row+mid_m*2, col+mid_n*2))

    for i in range(row-1):
        for j in range(col-1):
            temp_img[mid_m+i][mid_n+j] = img[i][j]
    
    convolved = np.zeros((row,col))

    for r in range(row):
        for c in range(col):
            convolved[r][c] = sum_of_products(temp_img, r+mid_m, c+mid_n,w)

    return convolved
# method to draw circles found by Hough Transform
def draw_circles(circles):
    if circles is not None:
        circles = np.round(circles[0,:]).astype('int')

        for(x,y,r) in circles:
            cv2.circle(output, (x,y), r, (220,40,0), 4)
            cv2.rectangle(output, (x-3, y-3), (x+3, y+3), (0,170,234), -1)
    
    return output
# generates a gaussian kernel 
def generate_kernel(size, sigma):
    size = size // 2

    row, col = np.mgrid[-size:size+1, -size:size+1]

    w = np.exp(-((row**2 + col**2) / (2.0*sigma**2))) * (1 / (2.0 * np.pi * sigma**2))

    return w
# method to apply a sobel filter to help create more defined edges
def sobel_filter(img):
    # matrices to be convolved with the img
    sobel_x = [[1,0,-1],[2,0,-2],[1,0,-1]]
    sobel_y = [[1,2,1],[0,0,0],[-1,-2,-1]]

    # convolve the img with sobel_x
    # convolve the img with sobel_y

original_img = cv2.imread("coins4.jpg")
# cv2.imshow("original", original_img)

# used to store the diameters of the detected circles
diameters = []
# circles will be drawn on this image
output = original_img.copy()

# convert image to gray-scale
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
row, col = np.shape(gray_img)

# gaussian blur
w = generate_kernel(9, 2.0)
blurred = convolve(gray_img, w)
blurred = np.uint8(blurred)
cv2.imshow("blurred", blurred)
cv2.waitKey(0)

# apply the sobel filter

# detect circles
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.5, 100, minRadius=20, maxRadius=100)
# draw circles on original image
draw_circles(circles)
        

# show the output image
cv2.imshow("output", output)
cv2.waitKey(0)



