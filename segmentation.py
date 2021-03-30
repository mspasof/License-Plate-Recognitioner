import cv2
import numpy as np
import os
from clusters import clusters
import csv

def histEqualization(image):
    result = image.copy()
    vals = np.zeros(256)
    for i in np.arange(len(result)):
        for j in np.arange(len(result[i])):
            ind = result[i][j]
            vals[ind] = vals[ind] + 1
    factor = len(image)*len(image[0])
    vals[0] = vals[0] / factor
    for i in np.arange(1, len(vals)):
        vals[i] = vals[i-1] + vals[i] / factor
    vals = vals * 255
    for i in np.arange(len(result)):
        for j in np.arange(len(result[i])):
            ind = result[i][j]
            result[i][j] = vals[ind]
    return result

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), 0)
        images.append(img)
    return images

def comparisonViaSSIM(image1, image2):
    nu_x = np.average(image1)
    nu_y = np.average(image2)
    var_x = np.var(image1)
    var_y = np.var(image2)
    covar = np.sum((image1 - np.mean(image1))*(image2 - np.mean(image2)))/(len(image1) - 1)
    c1 = np.square(0.01 * 255)
    c2 = np.square(0.03 * 255)
    SSIM = ((2 * nu_x * nu_y + c1) * (2 * covar + c2)) / ((np.square(nu_x) + np.square(nu_y) + c1) * (var_x + var_y + c2))
    return SSIM

def checkChar(image, path, bestMatch, char):
    images = load_images_from_folder('./data/' + str(path))
    for img in images:
        matched = comparisonViaSSIM(image, img)
        if matched > bestMatch:
            char = path
            bestMatch = matched
    return (bestMatch, char)


def bitWiseAllLetters(image):
    bestMatch = 0
    char = ''
    chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B', 'D', 'F',
             'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z']
    for character in chars:
        (bestMatch, char) = checkChar(image, character, bestMatch, char)
    return char, bestMatch

def numDifferentChars(newPlate, plateToCompare):
    diff = 0
    newPlate = newPlate.replace('-', '')
    plateToCompare = plateToCompare.replace('-', '')
    for i in range(len(plateToCompare)):
        if plateToCompare[i] != newPlate[i]:
            diff+=1
    return diff

# Function to extract segmentated characters from plate
def segmentation(plateImage, data, plates):
    plateImage = cv2.cvtColor(plateImage, cv2.COLOR_BGR2GRAY)
    if data[0] == data[2] and len(plates) > 0 and plates[0][1] > 2:
        with open('Output.csv', mode='a') as csv_file:
            if(plates[0][1] > 2):
                writer = csv.writer(csv_file)
                writer.writerow(["'" + plates[0][0] + "'", plates[0][2], plates[0][3]])
        return None
    resizedPlateImage = cv2.resize(plateImage, (140, 33))
    resizedPlateImage = histEqualization(resizedPlateImage)
    _, threshPlateImage = cv2.threshold(resizedPlateImage, 60, 255, cv2.THRESH_BINARY_INV)
    cl = clusters()
    for i in range(len(threshPlateImage)):
        for j in range(len(threshPlateImage[i])):
            if threshPlateImage[i][j] == 255:
                cl.addPoint((i, j))
    flag = cl.filterClusters()
    if flag == 'skip' or flag is None:
        return plates
    word = ''
    for i in cl.clusters:
        char = np.zeros((i.maxY - i.minY + 1, i.maxX - i.minX + 1))
        for j in i.points:
            char[j[0] - i.minY][j[1] - i.minX] = 255

        char = cv2.resize(src=char, dsize=(15, 15), interpolation=cv2.INTER_LINEAR)
        charToRecognize = np.zeros((25, 25))
        for i in range(15):
            for j in range(15):
                if(char[i][j] > 0):
                    charToRecognize[i + 5][j + 5] = 255
        charToRecognize = 255 - charToRecognize
        (char2, bestMatch) = bitWiseAllLetters(charToRecognize)
        word += char2
    newMatch = word[:flag[0]] + '-' + word[flag[0]:flag[1] - 1] + '-' + word[flag[1] - 1:]
    if len(newMatch) != 8:
        return plates
    if len(plates) == 0:
        plates.append([newMatch, 1, data[0], data[1]])
    elif int(numDifferentChars(newMatch, plates[0][0])) > 3:
        with open('Output.csv', mode='a') as csv_file:
            if(plates[0][1] > 2):
                writer = csv.writer(csv_file)
                writer.writerow(["'" + plates[0][0] + "'", plates[0][2], plates[0][3]])
        plates.clear()
        plates.append([newMatch, 1, data[0], data[1]])
    else:
        matched = False
        for i in range(len(plates)):
            if plates[i][0] == newMatch:
                matched = True
                plates[i][1] += 1
                plates = sorted(plates, key=lambda plates: plates[1], reverse=True)
                break
        if not matched:
            plates.append([newMatch, 1, data[0], data[1]])
    return plates
