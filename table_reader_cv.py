# PART 1: Header
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import csv

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
file = r'Capture.jpg'

# Part 2: Detecting Objects
im1 = cv2.imread(file, 0)
im = cv2.imread(file)
ret, thresh_value = cv2.threshold(im1, 100, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((5,5),np.uint8)
dilated_value = cv2.dilate(thresh_value,kernel,iterations = 1)

#cv2.imshow("image",dilated_value)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

contours, hierarchy = cv2.findContours(dilated_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
coordinates = []
coordinates_rectangle = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    coordinates.append((x, y, w, h))
    coordinates_rectangle.append((x,y,x+w,y+h))
    # bounding the images
    #if y < 2000:
    #    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)

# Part 3: Removing Non-cell Objects
subset_box = []
array_elements = []
buffer = 0
for cnt1 in range(1,len(coordinates_rectangle)):
    for cnt2 in range(1,len(coordinates_rectangle)):
        if (
        coordinates_rectangle[cnt1][0] > coordinates_rectangle[cnt2][0] and
        coordinates_rectangle[cnt1][1] > coordinates_rectangle[cnt2][1] and
        coordinates_rectangle[cnt1][2] < coordinates_rectangle[cnt2][2] and
        coordinates_rectangle[cnt1][3] < coordinates_rectangle[cnt2][3]):
            del coordinates[cnt1 - buffer]
            buffer = buffer + 1
            break


del coordinates[0]

#### Draw Rectangles if Desired ####
for cnt3 in coordinates:
    x = cnt3[0]
    y = cnt3[1]
    w = cnt3[2]
    h = cnt3[3]
    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 1)



plt.imshow(im)
cv2.namedWindow('table', cv2.WINDOW_NORMAL)
cv2.imwrite('table.jpg',im)

# Part 4: OCR
row = 0
col = 0
tbl_array = []
for cnt4 in range(0,len(coordinates)):
    x, y, w, h = coordinates[cnt4]
    cropped = im[y:(y + h), x:(x + w)]
    cv2.imshow("cropped", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(pytesseract.image_to_string(cropped))
    tbl_array.append(pytesseract.image_to_string(cropped).replace("\n"," "))
    # Get number of rows/columns
    if coordinates[0][0] == coordinates[cnt4][0]:
        row = row + 1
    if coordinates[0][1] == coordinates[cnt4][1]:
        col = col + 1

# Part 5: Write to table
output = np.zeros((row,col), dtype = object)
cell = 0
for cnt5 in range(0,row):
    for cnt6 in range(0,col):
        output[row - 1 - cnt5, col - 1 - cnt6] = tbl_array[cell]
        cell = cell + 1


np.savetxt("output.csv", output, delimiter=',', fmt='%s')







