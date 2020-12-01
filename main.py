import os
import sys

import random
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn import manifold

import time
import gc

import torchvision.models as models
import torch.nn as nn
import torch
import torch.optim as optim

import Project.fishLib as fishlib
from Project.ConstsAndHyperparameters import *

root_dir = os.getcwd()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if 'Load' in OPPERATING_MODES:
    modelVGG16 = torch.load(MODEL_DIR_PATH)
    os.chdir(root_dir)
else:
    modelResNet50 = models.resnet50(True, True)
    modelVGG16 = models.vgg16(True, True)
    modelVGG16.classifier[6] = nn.Linear(4096, EMBEDDING_SPACE)

    for layer in modelVGG16.features:
        layer.requires_grad = False

    for layer in modelVGG16.classifier:
        layer.requires_grad = True

print("Model ready")

data_label = os.listdir(DATA_DIR_PATH)

print("Number of images", len(data_label))
print(modelVGG16.classifier)
random.shuffle(data_label)

if 'Train' in OPPERATING_MODES:
    data, data_dir = fishlib.tripletConstruction(data_label, DATA_DIR_PATH, IMG_DIM, HARD_TRIPLETS)
    random.shuffle(data)
    print(data.size())
    split = int(len(data) * SPLIT_RATIO)
    data_train, data_val = data.split(split)
    print("Size of training set: ", len(data_train), "\n" +
          "Size of validation set: ", len(data_val))
    print("data_train", data_train.size())
else:
    data_dir = 'r'


tot_loss_val = []
tot_loss_train = []
tot_epoch = []

lossFunc = nn.TripletMarginLoss(TRIPLETLOSS_MARGIN)
optimizer = optim.SGD(modelVGG16.classifier.parameters(), lr=LEARNING_RATE, momentum=0.9)
modelVGG16.to(device)
print("Model loaded to GPU")

if 'Train' in OPPERATING_MODES:

    for epoce in range(EPOCHES):

        print("\n\n"+"="*40,
              "\nEpoce:", epoce,
              "\nHard triplets enabled:", HARD_TRIPLETS,
              "\nHard triplets set-size:", HARD_TRIPLET_RESTRICTION)

        tot_epoch.append(epoce + 1)
        for state in ['train', 'val']:
            if state == 'train':
                train_loss = 0
                for i in tqdm(range(0, len(data_train), BATCH_SIZE)):  #
                    batch = data_train[i:i + BATCH_SIZE]
                    batch_a = []
                    batch_p = []
                    batch_n = []
                    for tripllet in batch:
                        batch_a.append(np.asarray(tripllet[0]))
                        batch_p.append(np.asarray(tripllet[1]))
                        batch_n.append(np.asarray(tripllet[2]))

                    batch_a = torch.tensor(batch_a).to(device)
                    batch_p = torch.tensor(batch_p).to(device)
                    batch_n = torch.tensor(batch_n).to(device)

                    modelVGG16.zero_grad()

                    outputs_a = modelVGG16(batch_a)
                    outputs_p = modelVGG16(batch_p)
                    outputs_n = modelVGG16(batch_n)

                    loss = lossFunc(outputs_a, outputs_p, outputs_n)
                    loss.backward()
                    train_loss += loss.cpu().detach().numpy()

                    optimizer.step()

                tot_loss_train.append(train_loss / (len(data_train) / BATCH_SIZE))
                print("\n", f"Epoch: {epoce + 1}/{EPOCHES}. Loss: {train_loss / (len(data_train) / BATCH_SIZE)}")

            if state == 'val':
                val_loss = 0
                for i in (range(0, len(data_val), BATCH_SIZE)):
                    batch = data_val[i:i + BATCH_SIZE]
                    batch_a = []
                    batch_p = []
                    batch_n = []
                    for tripllet in batch:
                        batch_a.append(np.asarray(tripllet[0]))
                        batch_p.append(np.asarray(tripllet[1]))
                        batch_n.append(np.asarray(tripllet[2]))

                    batch_a = torch.tensor(batch_a).to(device)
                    batch_p = torch.tensor(batch_p).to(device)
                    batch_n = torch.tensor(batch_n).to(device)

                    outputs_a = modelVGG16(batch_a)
                    outputs_p = modelVGG16(batch_p)
                    outputs_n = modelVGG16(batch_n)

                    loss = lossFunc(outputs_a, outputs_p, outputs_n)
                    val_loss += loss.cpu().detach().numpy()
                tot_loss_val.append(val_loss / (len(data_val) / BATCH_SIZE))

        if epoce % 10 == 0 and not epoce == 0:
            past_loss = tot_loss_train[len(tot_loss_train)-10:]
            rellative_loss = (past_loss[0]-past_loss[9])/past_loss[0]

            if rellative_loss < RELLATIVE_LOSS_LIMIT:
                if HARD_TRIPLETS and HARD_TRIPLET_RESTRICTION > 3:
                    HARD_TRIPLET_RESTRICTION -= 1
                HARD_TRIPLETS = True

            if 'Save' in OPPERATING_MODES:
                os.chdir(root_dir)
                torch.save(modelVGG16, MODEL_DIR_PATH)
                print(" Model saved at: \n", MODEL_DIR_PATH)
            random.shuffle(data_label)
            data, data_dir = fishlib.tripletConstruction(data_label, DATA_DIR_PATH, IMG_DIM, HARD_TRIPLETS)

            if HARD_TRIPLETS:
                fishRegister = fishlib.faceRegister()
                test_set = []
                all_idVectores = []
                all_IDs = []
                corect_guess = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                false_guess = 0
                total_guess = 0

                all_guess = []

                print("\n\n", "=" * 40,
                      "\n Generating database")
                for ii in tqdm(range(0, len(data_label))):
                    label = data_label[ii]
                    ID, fish_dir, index = fishlib.fileNameAnalyzere(label)
                    if index > 1 and fish_dir == data_dir:
                        test_set.append(label)
                    elif fish_dir == data_dir:
                        img = fishlib.loadImage(label, DATA_DIR_PATH, IMG_DIM)
                        img = torch.tensor([img], dtype=torch.float).to(device)
                        idVector = modelVGG16(img)
                        all_idVectores.append(idVector.cpu().detach().numpy()[0])
                        all_IDs.append(ID)

                        fishRegister.addFace(ID, idVector.cpu().detach())

                print("\n\n", "=" * 40,
                      "\n Generating test score")
                for ii in tqdm(range(0, len(test_set))):
                    label = test_set[ii]
                    ID, fish_dir, __ = fishlib.fileNameAnalyzere(label)
                    if fish_dir == data_dir:
                        img = fishlib.loadImage(label, DATA_DIR_PATH, IMG_DIM)
                        img = torch.tensor([img], dtype=torch.float).to(device)
                        idVector = modelVGG16(img)
                        all_idVectores.append(idVector.cpu().detach().numpy()[0])
                        all_IDs.append(ID)

                        guess, tot_distance = fishRegister.imageGuess(idVector.cpu().detach())

                        all_guess.append([ID, [i[0] for i in guess]])
                        for jj in range(len(corect_guess)):
                            if ID == guess[jj][0]:
                                for qq in range(jj, len(corect_guess)):
                                    corect_guess[qq] += 1
                                break
                        total_guess += 1
                false_guess = total_guess - corect_guess[9]

                print(os.getcwd())
                f = open(HARD_TRIPLET_PATH, "w")
                for guess in all_guess:
                    f.write(str(guess[0]))
                    for value in guess[1]:
                        f.write("," + str(value))
                    f.write("\n")
                f.close()

                print("=" * 40,
                      "\nCorrect guesses:       ", corect_guess[0],
                      "\nTop 3 guesses:         ", corect_guess[2],
                      "\nTop 5 guesses:         ", corect_guess[4],
                      "\nTop 10 guesses:        ", corect_guess[9],
                      "\nFalse guesses:         ", false_guess,
                      "\nAccuracy top 1:         %1.3f" % (100 * corect_guess[0] / total_guess),
                      "\nAccuracy top 3:         %1.3f" % (100 * corect_guess[2] / total_guess),
                      "\nAccuracy top 5:         %1.3f" % (100 * corect_guess[4] / total_guess),
                      "\nAccuracy top 10:        %1.3f" % (100 * corect_guess[9] / total_guess),
                      "\nNumber of individes:   ", fishRegister.getNumberOfFaces(),
                      "\nRandom chance accurscy: %1.3f" % (100 / fishRegister.getNumberOfFaces()),
                      "\n" + "=" * 40)

                if epoce in [10, 50, 100]:
                    X = all_idVectores

                    Y = []
                    UsedID = []
                    for ii, ID in enumerate(all_IDs):
                        if not ID in UsedID:
                            Y.append(ii)
                            UsedID.append(ID)
                        else:
                            for jj in range(len(all_IDs)):
                                if all_IDs[jj] == ID:
                                    Y.append(jj)
                                    break

                    Y = np.array(Y)

                    fig = plt.figure()
                    plt.clf()

                    plt.cla()
                    tsne = manifold.TSNE(n_components=2)

                    X = tsne.fit_transform(X)

                    plt.scatter(X[:, 0], X[:, 1], c=Y)
                    plt.show()

if 'Save' in OPPERATING_MODES:
    os.chdir(root_dir)
    torch.save(modelVGG16, MODEL_DIR_PATH)
    print(" Model saved at: \n", MODEL_DIR_PATH)

if 'Plot' in OPPERATING_MODES and 'Train' in OPPERATING_MODES:
    plt.figure()
    val_patch = mpatches.Patch(color='blue', label='Validation loss')
    train_patch = mpatches.Patch(color='red', label='Training loss')
    plt.legend(handles=[val_patch, train_patch])

    plt.plot(tot_epoch, tot_loss_val, 'b', tot_epoch, tot_loss_train, 'r')
    plt.show()

if 'Validate' in OPPERATING_MODES:
    fishRegister = fishlib.faceRegister()
    test_set = []
    all_idVectores = []
    all_IDs = []
    corect_guess = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    false_guess = 0
    total_guess = 0

    all_guess = []

    print("\n\n", "=" * 40,
          "\n Generating database")
    for ii in tqdm(range(0, len(data_label))):
        label = data_label[ii]
        ID, fish_dir, index = fishlib.fileNameAnalyzere(label)
        if index > 1 and fish_dir == data_dir:
            test_set.append(label)
        elif fish_dir == data_dir:
            img = fishlib.loadImage(label, DATA_DIR_PATH, IMG_DIM)
            img = torch.tensor([img], dtype=torch.float).to(device)
            idVector = modelVGG16(img)
            all_idVectores.append(idVector.cpu().detach().numpy()[0])
            all_IDs.append(ID)

            fishRegister.addFace(ID, idVector.cpu().detach())

    print("\n\n", "=" * 40,
          "\n Generating test score")
    for ii in tqdm(range(0, len(test_set))):
        label = test_set[ii]
        ID, fish_dir, __ = fishlib.fileNameAnalyzere(label)
        if fish_dir == data_dir:
            img = fishlib.loadImage(label, DATA_DIR_PATH, IMG_DIM)
            img = torch.tensor([img], dtype=torch.float).to(device)
            idVector = modelVGG16(img)
            all_idVectores.append(idVector.cpu().detach().numpy()[0])
            all_IDs.append(ID)

            guess, tot_distance = fishRegister.imageGuess(idVector.cpu().detach())

            all_guess.append([ID, [i[0] for i in guess]])
            for jj in range(len(corect_guess)):
                if ID == guess[jj][0]:
                    for qq in range(jj, len(corect_guess)):
                        corect_guess[qq] += 1
                    break
            total_guess += 1
    false_guess = total_guess - corect_guess[9]

    print(os.getcwd())
    f = open(HARD_TRIPLET_PATH, "w")
    for guess in all_guess:
        f.write(str(guess[0]))
        for value in guess[1]:
            f.write("," + str(value))
        f.write("\n")
    f.close()


    print("=" * 40,
          "\nCorrect guesses:       ", corect_guess[0],
          "\nTop 3 guesses:         ", corect_guess[2],
          "\nTop 5 guesses:         ", corect_guess[4],
          "\nTop 10 guesses:        ", corect_guess[9],
          "\nFalse guesses:         ", false_guess,
          "\nAccuracy top 1:         %1.3f" % (100 * corect_guess[0] / total_guess),
          "\nAccuracy top 3:         %1.3f" % (100 * corect_guess[2] / total_guess),
          "\nAccuracy top 5:         %1.3f" % (100 * corect_guess[4] / total_guess),
          "\nAccuracy top 10:        %1.3f" % (100 * corect_guess[9] / total_guess),
          "\nNumber of individes:   ", fishRegister.getNumberOfFaces(),
          "\nRandom chance accurscy: %1.3f" % (100 / fishRegister.getNumberOfFaces()),
          "\n" + "=" * 40)

    X = all_idVectores

    Y = []
    UsedID = []
    for ii, ID in enumerate(all_IDs):
        if not ID in UsedID:
            Y.append(ii)
            UsedID.append(ID)
        else:
            for jj in range(len(all_IDs)):
                if all_IDs[jj] == ID:
                    Y.append(jj)
                    break

    Y = np.array(Y)

    fig = plt.figure()
    plt.clf()

    plt.cla()
    tsne = manifold.TSNE(n_components=2)

    X = tsne.fit_transform(X)

    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.show()
