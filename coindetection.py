"""
Brenden Blake
CS 456
Project

Using image processing techniques to help in the detection of American coins
"""
import numpy as np
import cv2
import pandas as pd

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
    diameters = []
    if circles is not None:
        circles = np.round(circles[0,:]).astype('int')

        for(x,y,r) in circles:
            cv2.circle(output, (x,y), r, (220,40,0), 4)
            cv2.rectangle(output, (x-3, y-3), (x+3, y+3), (0,170,234), -1)
            diameters.append(r*2)
    
    return diameters

# generates a gaussian kernel 
def generate_kernel(size, sigma):
    size = size // 2

    row, col = np.mgrid[-size:size+1, -size:size+1]

    w = np.exp(-((row**2 + col**2) / (2.0*sigma**2))) * (1 / (2.0 * np.pi * sigma**2))

    return w

# method to apply a sobel filter to help create more defined edges
def sobel_filter(img):
    # matrices to be convolved with the img
    sobel_y = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    sobel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    # convolve the img with horizontal kernel
    img_y = cv2.filter2D(img, -1, kernel=sobel_x)

    # convolve the new image with the vertical kernel
    final_img = cv2.filter2D(img_y, -1, kernel=sobel_y)

    return final_img

if __name__ == '__main__':
    original_img = cv2.imread("coins3.jpg")
    row, col, third = np.shape(original_img)

    if row < 748 or row > 748:
        # increase/decrease the size of the image
        scaler = (748 / row) * 100
        row = int(row * scaler / 100)
        if col < 748 or col > 748:
            scaler = (748 / col) * 100
            col = int(col * scaler / 100)
        else:
            col = int(col * scaler / 100) 
        original_img = cv2.resize(original_img, (col, row), fx=2, fy=2)
        print(row, col)


    # convert image to gray-scale
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # apply the sobel filter
    sobel = sobel_filter(gray_img)
    # cv2.imshow("sobel", sobel)
    # cv2.imwrite("sobel_filt.png", sobel)
    # cv2.waitKey(0)

    # dilate the image
    dilate_kernel = np.ones((7,7))
    dilated = cv2.dilate(sobel, dilate_kernel)
    # cv2.imshow("Dilated", dilated)
    # cv2.waitKey(0)

    closed_img = cv2.erode(dilated, dilate_kernel)
    # cv2.imshow("closed", closed_img)
    # cv2.imwrite("closed_img.png", closed_img)
    # cv2.waitKey(0)

    # gaussian blur
    w = generate_kernel(3, 2.0)
    blurred = cv2.GaussianBlur(closed_img, (5,5), cv2.BORDER_DEFAULT)
    # cv2.imwrite("blurred.png", blurred)
    # cv2.imshow("closed blurred", blurred)
    # cv2.waitKey(0)

    # circles will be drawn on this image
    output = original_img.copy()

    # detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.6, 50, 30,  minRadius=10, maxRadius=130)
    # draw circles on original image and store the diameters in an array
    diameters = draw_circles(circles)

    # determine the ranges of the coins' diameters
    length = len(diameters)
    quarter = max(diameters)
    dime = min(diameters)
    median = int(np.median(diameters))
    penny = int(np.median(diameters[length//2:length]))
    nickel = int(np.median(diameters[0:(length//2)]))

    quarter_range = (quarter - 4, quarter)
    nickel_range = (nickel-8, nickel + 3)
    penny_range = (penny-2, penny)
    dime_range = (dime, dime + 4)

    # decide which denomination each coin is and count each denomination
    quarter_amt = 0
    nickel_amt = 0
    penny_amt = 0
    dime_amt = 0

    for diameter in diameters:
        if diameter >= quarter_range[0]:
            quarter_amt += 1
        elif diameter >= nickel_range[0] and diameter < quarter_range[0]:
            nickel_amt += 1
        elif diameter >= penny_range[0] and diameter < nickel_range[0]:
            penny_amt += 1
        else:
            dime_amt += 1

    # create the output table
    table_row_label = ["Quarter", "Nickel", "Penny", "Dime"]
    tabel_col_label = ["Amount", "Dollar Amount"]
    data = {"Amount": [quarter_amt, nickel_amt, penny_amt, dime_amt], "Dollar Amount": [quarter_amt*0.25, nickel_amt*0.05, penny_amt*0.01, dime_amt*0.1]}
    table = pd.DataFrame(data, index = table_row_label)
    print("\n", table)
    print("\nTotal: $", sum(table["Dollar Amount"]))
    # show the output image
    cv2.imshow("output", output)
    cv2.imwrite("output_img.png", output)
    cv2.waitKey(0)

