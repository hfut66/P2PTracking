import cv2
import numpy as np
import csv
import time
from statistics import mean
import csv
import os

def get_groundTruth(path):
    with open(path,'r') as f:
        reader = csv.reader(f)
        next(reader)
        labels = [[float(i[0]),float(i[1])] for i in reader ]
        return labels

def get_trackpoints(path):
    f = np.loadtxt(path, delimiter=' ')
    p = np.zeros((f.shape[0], 2))
    for i in range(f.shape[0]):
        p[i][0] = f[i][0]
        p[i][1] = f[i][1]
    return p

def MSE_evaluate(tracked_points, labelsNew):
    MSE_result = []
    # for i in range(len(tracked_points)):
    for i in range(149):
        mse_dis = np.power(labelsNew[i+1][0] - tracked_points[i][0], 2) + np.power(labelsNew[i+1][1] - tracked_points[i][1],2)
        MSE_result.append(mse_dis)
    MSE = np.sum(MSE_result) / len(tracked_points)
    return MSE

def Euclidean_evaluate(tracked_points, labelsNew):
    Euclidean_list = []
    for i in range(len(tracked_points)):
        dis = np.sqrt(np.power(labelsNew[i+1][0]-tracked_points[i][0],2)+np.power(labelsNew[i+1][1]-tracked_points[i][1],2))
        Euclidean_list.append(dis)
    Euclidean= np.sum(Euclidean_list)/len(tracked_points)
    return Euclidean

def AR_evaluate(tracked_points_path, labels_path):
    labelsNew=get_groundTruth(labels_path)
    tracked_points = get_trackpoints(tracked_points_path)
    print(labelsNew[0])
    pos1 = 0
    pos2 = 0
    pos3 = 0
    pos4 = 0
    pos5 = 0
    pos6 = 0
    pos7 = 0
    pos8 = 0

    for i in range(len(tracked_points)):
        dis = np.sqrt(np.power(labelsNew[i+1][0]-tracked_points[i][0],2)+np.power(labelsNew[i+1][1]-tracked_points[i][1],2))
        if dis <=1:
            pos1+=1
        if dis <=2:
            pos2+=1
        if dis <=3:
            pos3+=1
        if dis <=4:
            pos4+=1
        if dis <=5:
            pos5+=1
        if dis <= 6:
            pos6 += 1
        if dis <= 7:
            pos7 += 1
        if dis <= 8:
            pos8 += 1
    AR1 = pos1 / len(tracked_points)
    AR2 = pos2 / len(tracked_points)
    AR3 = pos3 / len(tracked_points)
    AR4 = pos4 / len(tracked_points)
    AR5 = pos5 / len(tracked_points)
    AR6 = pos6/len(tracked_points)
    AR7 = pos7/ len(tracked_points)
    AR8 = pos8/ len(tracked_points)
    mse=MSE_evaluate(tracked_points, labelsNew)
    Euclidean=Euclidean_evaluate(tracked_points, labelsNew)
    return AR1,AR2,AR3,AR4,AR5,AR6,AR7,AR8,mse,Euclidean

label_path="./Point_Annotations/csv/video1_point.csv"
tracked_points_path="path to your txt"
a1,a2,a3,a4,a5,a6,a7,a8,mse,Euclidean=AR_evaluate(tracked_points_path, label_path)
Result_pattern = ['path={},AR1 = {},AR2 = {},AR3 = {},AR4 = {},AR5 = {},AR6 = {},AR7 = {},AR8 = {},Distance = {},MSE = {}'.format(tracked_points_path,a1,a2,a3,a4,a5,a6,a7,a8,mse,Euclidean)]
print('Result_pattern = ', Result_pattern)

file = open('path to your txt', 'a')
file.write(str(Result_pattern) + '\n')
file.close()


print(a2)
print(a3)
print(a4)
print(a5)
print(mse)
print(Euclidean)




