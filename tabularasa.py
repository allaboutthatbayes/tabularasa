# PART 1: Header
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from sklearn.cluster import KMeans
import pandas as pd
import math
import re
import os
import csv
from operator import itemgetter

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Select image from directory
directory = os.listdir()
r = re.compile("([^\\s]+(\\.(?i)(jpg|png|gif|bmp))$)") #show only image files
png_files = list(filter(r.match, directory))
print("The following image files were found in the directory:")
for i in range(0,len(png_files)):
    print(str(i+1) + ". " + png_files[i])
print("Type image number to begin OCR:")
value = int(input())
file = png_files[value-1]


# Part 2: Preliminary Analysis and Transformation
# Part 2.1: Read Image
im1 = cv2.imread(file, 0) # Read Image
im = cv2.imread(file)
ret, binarised = cv2.threshold(im1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) # Binarise Image

# Part 2.2: Trim whitespace around table and remove lines
# Trim vertical whitespace at start
i = 0
while i == 0:
    i = sum(binarised[:,0])
    if i > 0:
        break
    binarised = np.delete(binarised, 0, axis = 1)

# Trim vertical whitespace at end
i = 0
while i == 0:
    i = sum(binarised[:,len(binarised[0])-1])
    if i > 0:
        break
    binarised = np.delete(binarised, len(binarised[0])-1, axis = 1)

#Trim horizontal whitespace at start
j = 0
while j == 0:
    j = sum(binarised[0,:])
    if j > 0:
        break
    binarised = np.delete(binarised, 0, axis = 0)

#Trim horizontal whitespace at end
j = 0
while j == 0:
    j = sum(binarised[len(binarised)-1,:])
    if j > 0:
        break
    binarised = np.delete(binarised, len(binarised)-1, axis = 0)

#Add border
binarised = cv2.copyMakeBorder(binarised, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value= [0,0,0])

#Remove Lines
col_lines = []
row_lines = []

for i in range(0, binarised.shape[1]):
    colpct = sum(binarised[:,i]) / (255 * binarised.shape[0])  # find percentage of col filled
    if colpct > 0.8:
        col_lines.append(i)

for j in range(0, binarised.shape[0]):
    colpct = sum(binarised[j, :]) / (255 * binarised.shape[1])  # find percentage of col filled
    if colpct > 0.8:
        row_lines.append(j)

#Removes lines from image
binarised = np.delete(binarised, col_lines, axis = 1)
binarised = np.delete(binarised, row_lines, axis = 0)


# Part 2.3: Average Feature Size
contours, hierarchy = cv2.findContours(binarised, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
features = np.empty((0,4), int)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    features = np.append(features,
                         np.array([[x,y,w,h]]),
                         axis=0)
avg_width = int(round(sum(features[:,2])/len(features)))
avg_height = int(round(sum(features[:,3])/len(features)))
print("Average character size is " + str(avg_width) + " by " + str(avg_height) + " pixels")

# Part 2.4: Image Preprocessing
if avg_width*avg_height < 900:
    print("Image too small, resizing to minimum character size of 25 by 25 pixels...")
    scale = 30/avg_width
    binarised = cv2.resize(binarised, (0,0), fx = scale, fy = scale, interpolation=cv2.INTER_LANCZOS4)
    binarised = cv2.medianBlur(binarised, 3)


#test = cv2.medianBlur(binarised, 7)
#cv2.imshow("cropped", test)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()


# Part 2.5: Image Dilation
kernel = np.ones((25,25),np.uint8) #dilation kernel size
dilated_value = cv2.dilate(binarised, kernel,iterations = 1)
scale = 1280/dilated_value.shape[1]
preview = cv2.resize(dilated_value, (0,0), fx = scale, fy = scale)
cv2.imshow("image", preview)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Part 2.6: Create horizontal and vertical histograms
row_hist = []
col_hist = []
#columns
for i in range(0,dilated_value.shape[1]):
    col_hist.append(sum(dilated_value[:,i]))

for j in range(dilated_value.shape[0]):
    row_hist.append(sum(dilated_value[j,:]))

row_hist = np.array(row_hist)
col_hist = np.array(col_hist)

#plots
plt.bar(range(0,len(row_hist)), height = row_hist)
plt.show()
plt.clf
plt.bar(range(0,len(col_hist)), height = col_hist)
plt.show()


# Part 3: K-Means Clustering of Projection Histogram Minima
print("Clustering minima and maxima of projection histograms...")
# Perform clustering to identify table minima
zeros = np.zeros(len(row_hist))
cluster_row = np.empty((len(row_hist),3),int)
cluster_row[:,0] = row_hist
cluster_row[:,1] = zeros
km = KMeans(n_clusters=2)
cluster_row[:,2] = km.fit_predict(cluster_row[:,0:1])

zeros = np.zeros(len(col_hist))
cluster_col = np.empty((len(col_hist),3),int)
cluster_col[:, 0] = col_hist
cluster_col[:, 1] = zeros
km = KMeans(n_clusters=2)
cluster_col[:, 2] = km.fit_predict(cluster_col[:,0:1])


# Part 4: Calculate Widths of Candidate Column and Row Boundaries
# Rows
n = cluster_row[0][2]
m = cluster_col[0][2]
candidate_row = np.empty((0,4), int)
start = 0
for i in range(0,len(cluster_row)-1):
    if cluster_row[i, 2] != n and cluster_row[i, 2] != cluster_row[i+1, 2]:
        start = i
    if cluster_row[i, 2] == n and cluster_row[i, 2] != cluster_row[i+1, 2]:
        temp = cluster_row[start:i,0]
        index_min = start + 5 + min(range(len(temp)),
                                    key=temp.__getitem__)
        candidate_row = np.append(candidate_row,
                                  np.array([[start, i, index_min, i - start]]),
                                  axis = 0) #(start,end,midpoint, length)

candidate_row = np.append(candidate_row, np.array([[0,0,dilated_value.shape[0]-35,100]]), axis = 0)

# Columns
candidate_col = np.empty((0,4), int)
start = 0
for i in range(0,len(cluster_col)-1):
    if cluster_col[i, 2] != m and cluster_col[i, 2] != cluster_col[i+1, 2]:
        start = i
    if cluster_col[i, 2] == m and cluster_col[i, 2] != cluster_col[i+1, 2]:
        temp = cluster_col[start:i, 0]
        index_min = start +  5 + min(range(len(temp)),
                                     key=temp.__getitem__)
        candidate_col = np.append(candidate_col,
                                  np.array([[start, i, index_min, i - start]]),
                                  axis = 0) #(start,end,line start, length)
candidate_col = np.append(candidate_col, np.array([[0,0,dilated_value.shape[1]-35,100]]), axis = 0)

# Part 5: Exclude Spurious Minima and Maxima
print("Excluding spurious minima and maxima...")
# Remove intervals that are less than 20 pixels in width
candidate_col = candidate_col[np.where(candidate_col[:,3] > 20)]
candidate_row = candidate_row[np.where(candidate_row[:,3] > 20)]


# Draw Lines
lined = cv2.cvtColor(binarised, cv2.COLOR_GRAY2BGR)
scale = 1280/lined.shape[1]
lined = cv2.resize(lined, (0,0), fx = scale, fy = scale)
for i in range(0, len(candidate_col)):
    cv2.line(lined,
             (int(candidate_col[i,2]*scale), 0),
             (int(candidate_col[i,2]*scale), lined.shape[0]),
             (0,0,255), 1)
for i in range(0, len(candidate_row)):
    cv2.line(lined,
             (0, int(candidate_row[i,2]*scale)),
             (lined.shape[1], int(candidate_row[i,2]*scale)),
             (0,0,255), 1)

cv2.imshow("cropped", lined)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Part 6: OCR and Output
print("Running OCR...")
row_coords = np.round(candidate_row[:,2]).astype(int)
col_coords = np.round(candidate_col[:,2]).astype(int)
row_coords[0] = 0
col_coords[0] = 0
row_coords[len(row_coords)-1] = binarised.shape[0]
col_coords[len(col_coords)-1] = binarised.shape[1]

output = np.zeros((len(row_coords)-1,len(col_coords)-1),dtype = object)
for i in range(0,len(row_coords)-1):
    for j in range(0, len(col_coords)-1):
        cropped = binarised[row_coords[i]: row_coords[i+1], col_coords[j]:col_coords[j+1]]
        dilated = cv2.dilate(cropped, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            cropped = cropped[y:(y + h), x:(x + w)]
        #cv2.imshow("cropped", cropped)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        data = pytesseract.image_to_data(cropped, output_type='data.frame')
        if max(data['conf']) > 80:
            output[i][j] = pytesseract.image_to_string(cropped)
            print(output[i][j])
        else:
            output[i][j] = ''

df = pd.DataFrame(output)
df.to_excel("output.xlsx")



