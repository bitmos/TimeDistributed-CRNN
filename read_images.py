# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:44:41 2019

@author: Gireesh Sundaram
"""

import cv2
import numpy as np
import pandas as pd
import codecs
import math

from configuration import window_height, window_width, window_shift, MPoolLayers_H, nb_labels

from keras.preprocessing import sequence

#%%
#reading the class files
data = {}
with codecs.open("Data/class.txt", 'r', encoding='utf-8') as cF:
    data = cF.read().split('\n')
    
#%%
def returnClasses(string):
    text = list(string)
    text = ["<SPACE>"] + ["<SPACE>" if x==" " else x for x in text] + ["<SPACE>"]
    classes = [data.index(x) if x in data else 2 for x in text]
    classes = np.asarray(classes)
    return classes
    

#%%
base_image_path = os.path.join(base_path, "sentences")
def get_image_paths_and_labels(samples):
    paths = []
    corrected_samples = []
    for (i, file_line) in enumerate(samples):
        line_split = file_line.strip()
        line_split = line_split.split(" ")

        # Each line split will have this format for the corresponding image:
        # part1/part1-part2/part1-part2-part3.png
        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path = os.path.join(
            base_image_path, partI, partI + "-" + partII, image_name + ".png"
        )
        if os.path.getsize(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])

    return paths, corrected_samples



def get_labels(train_labels):
    train_labels_cleaned = []
    characters = set()
    max_len = 0
    for label in train_labels:
        label = label.split(" ")[-1].strip()
        label=label.split("|")
        label=" ".join(label)
    
    for char in label:
        characters.add(char)

    max_len = max(max_len, len(label))
    train_labels_cleaned.append(label)

    return train_labels_cleaned

def find_max_width(paths):
    max_width = 0
    for record in range(0, len(paths)):
        
        image = cv2.imread(record, cv2.IMREAD_GRAYSCALE)
        
        h, w = np.shape(image)
        
        if (h > window_height): factor = window_height/h
        else: factor = 1
        
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
        h, w = np.shape(image)
        
        if w / window_width < math.ceil(w / window_width):
            padding = np.full((window_height, math.ceil(w / window_width) * 64 - w), 255)
            image = np.append(image, padding, axis = 1)
        
        h, w = np.shape(image)
        if w > max_width: max_width = w
    return(max_width)
        
#%%
# 

def split_frames(path,max_width):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    h, w = np.shape(image)
    
    if (h > window_height): factor = window_height/h
    else: factor = 1
    
    image = cv2.resize(image, None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
    h, w = np.shape(image)
    
    if w / window_width < math.ceil(w / window_width):
        padding = np.full((window_height, math.ceil(w / window_width) * 64 - w), 255)
        image = np.append(image, padding, axis = 1)
    
    h, w = np.shape(image)
    frames = np.full((max_width // window_width, window_height, window_width, 1), 255)
    
    for slide in range(0, w // window_width):
        this_frame = image[:, slide * window_width : (window_width) * (slide+1)]
        this_frame = np.expand_dims(this_frame, 2)
        frames[slide] = this_frame
        
    return frames

#%%
def prepareData(path,label,max_width):

    x_train = np.zeros((len(infile), max_width // window_width, window_height, window_width, 1))    
    y_train = []
    im_train = []
    
    for record in range(0, path):
        print("Reading file: " + str (record))
        path = path[record]
        annotation = label[record]
        
        image = cv2.imread(record, cv2.IMREAD_GRAYSCALE)
        
        h, w = np.shape(image)
        
        if (h > window_height): factor = window_height/h
        else: factor = 1
        
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC).T
        h, w = np.shape(image)
        
        im_train.append(image)
        y_train.append(returnClasses(annotation))
        
        x_train_len = np.asarray([len(im_train[i]) for i in range(len(im_train))])
        x_train_len = (x_train_len/18).astype(int)
        y_train_len = np.asarray([len(y_train[i]) for i in range(len(y_train))])
        
        x_train[record] = split_frames(path,max_width)
        
        y_train_pad = sequence.pad_sequences(y_train, value=float(nb_labels), dtype='float32', padding="post")

    return x_train, y_train_pad, x_train_len, y_train_len