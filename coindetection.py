import numpy as np
import cv2
import scipy.stats as st


# threshhold method, accepts an img and a lower bound
def threshold(img, lower):
    [row,col] = np.shape(img)
    threshold_img = np.zeros((row,col))
    threshold_img = np.uint8(threshold_img)

    for i in range(row):
        for j in range(col):
            if gray_img[i][j] > lower:
                threshold_img[i][j] = 255
            else:
                threshold_img[i][j] = 0

    return threshold_img

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

def blur_img(img, w):
    row,col = np.shape(img)
    m,n = np.shape(w)

    w = np.rot90(w, 2)
    mid_m = m // 2
    mid_n = n // 2
    temp_img = np.zeros((row+mid_m*2, col+mid_n*2))

    for i in range(row-1):
        for j in range(col-1):
            temp_img[mid_m+i][mid_n+j] = img[i][j]
    
    blurred = np.zeros((row,col))

    for r in range(row):
        for c in range(col):
            blurred[r][c] = sum_of_products(temp_img, r+mid_m, c+mid_n,w)

    return blurred

def draw_circles(circles):
    if circles is not None:
        circles = np.round(circles[0,:]).astype('int')

    for(x,y,r) in circles:
        cv2.circle(output, (x,y), r, (0,255,0), 4)
        cv2.rectangle(output, (x-5, y-5), (x+5, y+5), (0,128,255), -1)
    
    return output


original_img = cv2.imread("coins4.jpg")
# cv2.imshow("original", original_img)

diameters = []
output = original_img.copy()

gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
row, col = np.shape(gray_img)

w = np.ones((7,7))
w_sum = w.sum()
w = 1 / (w_sum*w)

# threshold_img = threshold(gray_img, 120)

circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1.2, 100, minRadius=15, maxRadius=100)
draw_circles(circles)
        


cv2.imshow("output", output)
cv2.waitKey(0)   

# # cv2.imwrite("reddots.jpg", output)

# print(diameters)

# sum = 0.0
# for i in range(len(diameters)):
#     if(diameters[i] >= 120):
#         print("Quarter")
#         sum += 0.25
#     elif(diameters[i] < 120 and diameters[i] >= 110):
#         print("Nickel")
#         sum += 0.05

# print("Total amount is: ${:0.2f}".format(sum))

