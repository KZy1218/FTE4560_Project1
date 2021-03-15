import cv2
import os
import numpy as np
path_name = '/Users/macpro/Desktop/FTE4560_Project1/new-images'


def readImages(path_name):
    images = dict()
    labels = dict()
    for item in os.listdir(path_name):
        if item != '.DS_Store':
            label = item.split('.')[1]
            subject = item.split('.')[0]

            img = cv2.imread(path_name + '/' + item, cv2.IMREAD_GRAYSCALE)
            new_img = list(map(lambda x: x / 255, img.flatten()))
            if subject in images:
                images[subject].append(new_img)
                labels[subject].append(label)
            else:
                images[subject] = [new_img]
                labels[subject] = [label]

    return labels, images
            

labels, images = readImages(path_name)

# 获取所有subject
keys = list(labels.keys())
keys.sort()







