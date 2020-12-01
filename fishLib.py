import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm

from Project.ConstsAndHyperparameters import *

from PIL import Image
import pandas as pd
import numpy as np
import random

import os
import sys

def fileNameAnalyzere(fileName):
    ID, direction, index = fileName.split('_')
    index, _ = index.split('.')
    return int(ID), direction, int(index)

def filesToVector(files):
    x = []
    for file in files:
        x.append([fileNameAnalyzere(file), file])
    return x

def directionSorter(setOfFiles):
    left, right = [], []
    for file in setOfFiles:
        _, direction, __ = fileNameAnalyzere(file)
        if direction == 'l' or direction == 'L':
            left.append(file)
        else:
            right.append(file)
    return left, right

def findFileOfSameFish(file, files):
    sameFish = []
    for f in files:
        if f[0][0] == file[0][0]:
            sameFish.append(f)
    for f in sameFish:
        if f[1] == file[1]:
            sameFish.remove(f)
    if len(sameFish) == 0:
        return -1
    return sameFish[random.randint(0, len(sameFish) - 1)]

def loadImage(filename, dirpath, img_size):
    img = Image.open(dirpath+filename).resize(img_size)
    img = np.einsum('kji->ijk', np.array(img, dtype=np.float))
    return img/255

def tripletConstruction(files, filedir, img_size, hard_triplets = False):
    listOfTriplets = []

    if hard_triplets:
        data = pd.read_csv(HARD_TRIPLET_PATH, sep=',')
        data = np.asarray(data)
        hard_triplets = []
        for d in data:
            hard_triplets.append([d[0], d[1:HARD_TRIPLET_RESTRICTION+1]])

    left, right = directionSorter(files)

    if len(left) > len(right):
        list_of_files = filesToVector(left)
        dir = 'l'
    else:
        list_of_files = filesToVector(right)
        dir = 'r'

    for file in list_of_files:
        file_pos = findFileOfSameFish(file, list_of_files)
        if file_pos == -1:
            break
        file_neg = file
        while file_neg[0][0] == file[0][0]: # or file_neg == file_pos:
            if not hard_triplets:
                file_neg = random.choices(list_of_files)[0]
            else:
                for ii in range(len(hard_triplets)):
                    if file[0][0] == hard_triplets[:][ii][0]:
                        hard_ids = hard_triplets[:][ii][1]
                        break
                hard_list_of_files = []
                for hard_file in list_of_files:
                    if hard_file[0][0] in hard_ids and not hard_file[0][0] == file[0][0]:
                        hard_list_of_files.append(hard_file)
                file_neg = random.choices(hard_list_of_files)[0]

        listOfTriplets.append([loadImage(file[1], filedir, img_size),
                               loadImage(file_pos[1], filedir, img_size),
                               loadImage(file_neg[1], filedir, img_size)])
    return torch.tensor(listOfTriplets, dtype=torch.float), dir

def euclidianDistance(a, b):
    return (a - b).pow(2).sum().pow(0.5)

class faceRegister:
    def __init__(self):
        self.register = []

    def addFace(self, id, idVector):
        for instance in self.register:
            if instance[0] == id:
                return
        self.register.append([id, idVector])

    def imageGuess(self, idVector):
        best = []
        tot_distance = 0
        for instance in self.register:
            disntance = euclidianDistance(idVector, instance[1])
            tot_distance += disntance
            best.append([instance[0], disntance])
        best.sort(key=lambda x: x[1])
        return best[:10], tot_distance

    def getNumberOfFaces(self):
        return len(self.register)

def displayTriplet(trip):
    img1 = Image.fromarray(trip[0], 'RGB')
    img2 = Image.fromarray(trip[1], 'RGB')
    img3 = Image.fromarray(trip[2], 'RGB')
    dst = Image.new('RGB', (img1.width + img2.width + img3.width, img1.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (img1.width, 0))
    dst.paste(img3, (img2.width + img1.width, 0))
    dst.show()