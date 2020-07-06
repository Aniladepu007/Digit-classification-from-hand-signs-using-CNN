import os
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def showIm(img):
    plt.axis('off')
    plt.imshow(img,cmap='gray')
    plt.show()

def return_list(data_path, data_type):
    file_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
    return file_list

data_type = '.jpeg'
data_img_path = './dataset/'

unsegmentedImages_400x400 = []

for i in range(0,10):
    temp = data_img_path
    temp += (str(i) + '/')
    file_list = return_list(temp, data_type)

    for idX in range(len(file_list)):
        print("processing the image : {tmp}".format(tmp=(i+1)*(idX+1)))
        fileName = file_list[idX]
        # org_img = cv2.imread(temp + fileName)
        # grayImg = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
        org_img = np.asarray(image.load_img(temp + fileName, color_mode = 'grayscale'))

        unsegmentedImages_400x400.append(org_img)


np.save('unsegmentedImages_grayScale_400x400',np.asarray(segmentedImages_400x400))
