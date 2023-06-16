import cv2
import numpy as np
import torch
import argparse
from siamfc import TrackerSiamFC
# import math
# from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import os
import csv
import time
from statistics import mean
# import joblib

def parse_args():
    """
    args for testing.
    """
    parser = argparse.ArgumentParser(description='PyTorch SiamFC Tracking Demo')
    parser.add_argument('--model', dest='model', default='./checkPoint/siamfc/model.pth', help='pretrained model')
    parser.add_argument('--images', dest='images', default='./Point_Annotations/images/video1', help='video path')
    parser.add_argument('--labels',dest = 'labels', default='./Point_Annotations/csv/video1_point.csv', help='label path')
    # 跟踪框大小
    parser.add_argument('--boxSize',dest = 'boxSize',default=[100,150,200,250,300,'No_Patch_tracking'])
    parser.add_argument('--boxLegendName',dest = 'boxLegendName',default=['s=100','s=150','s=200','s=250','s=300','No_Patch_tracking'],help= 'boxLegend title')

    # 有无模板更新及阈值
    parser.add_argument('--upDatePattern', dest='upDatePattern', default=['No_update', 5, 10, 15])
    parser.add_argument('--upDateLegendName', dest='upDateLegendName', default=['No_update', '5', '10', '15'], help='upDatePattern title')
    parser.add_argument('--LegendColor',dest = 'LegendColor',default=['orange','black','blue','green','red','purple','deeppink'],help ='LegendColor')
    parser.add_argument('--LegendMarker', dest='LegendMarker',default=['s', 'x', '+', '*'],help='LegendMarker')
    parser.add_argument('--LegendStyle', dest='LegendStyle', default=['solid', 'dashed', 'dotted'],help='LegendStyle')
    args = parser.parse_args()

    return args

def _x1y1wh_to_xyxy(bbox_x1y1wh):
    x1, y1, w, h = bbox_x1y1wh
    x2 = x1+w
    y2 = y1+h
    return x1, y1, x2, y2

# 读取所有labels
def get_groundTruth(path):
    with open(path,'r') as f:
        reader = csv.reader(f)
        next(reader)
        labels = [[float(i[0]),float(i[1])] for i in reader]

        return labels

#性能评估
def AR_evaluate(tracked_points, labelsNew): #准确率评估---欧式距离
    pos5 = 0
    pos6 = 0
    pos7 = 0
    for i in range(len(labelsNew)):
        dis = np.sqrt(np.power(labelsNew[i][0]-tracked_points[i][0],2)+np.power(labelsNew[i][1]-tracked_points[i][1],2))
        if dis <=1:
            pos5+=1
        if dis <=3:
            pos6+=1
        if dis <=5:
            pos7+=1
    AR5 = pos5/len(labelsNew)
    AR6 = pos6/ len(labelsNew)
    AR7 = pos7/ len(labelsNew)

    return AR5,AR6,AR7

def MSE_evaluate(tracked_points, labelsNew): # MSE评估
    MSE_result = []
    for i in range(len(labelsNew)):
        mse_dis = np.power(labelsNew[i][0] - tracked_points[i][0], 2) + np.power(labelsNew[i][1] - tracked_points[i][1],2)
        MSE_result.append(mse_dis)
    MSE = np.sum(MSE_result) / len(labelsNew)

    return MSE

def Euclidean_evaluate(tracked_points, labelsNew): # 欧氏距离评估
    Euclidean_list = []
    for i in range(len(labelsNew)):
        dis = np.sqrt(np.power(labelsNew[i][0]-tracked_points[i][0],2)+np.power(labelsNew[i][1]-tracked_points[i][1],2))
        Euclidean_list.append(dis)
    Euclidean= np.sum(Euclidean_list)/len(labelsNew)

    return Euclidean


def main(args):
    # 初始所有图片
    dirs_init = os.listdir(args.images)  # dirs =  ['1.jpg', '10.jpg', '100.jpg', '101.jpg',...,]
    dirs_init.sort(key=lambda x: int(x[:-4]))  # dirs =  ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg',...,]
    # 初始所有标签
    labels_init = get_groundTruth(args.labels)
    print("dirs_init 长度= ", len(dirs_init))
    print('labels_init 长度 = ', len(labels_init))

    # 跟踪器初始化
    trk = TrackerSiamFC(net_path=args.model)
    ## 初始化特征检测器
    sift = cv2.SIFT_create()
    # sift = cv2.xfeatures2d.SIFT_create()
    # surf = cv2.xfeatures2d.SURF_create()
    surf = cv2.SURF_create()
    # 特征匹配初始化
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # KD树个数：5
    search_params = dict(checks=50)  # 遍历次数
    flann = cv2.FlannBasedMatcher(index_params, search_params)  # 创建匹配器

    while(True):
        # 画布
        plt.figure(dpi=900)
        # 修改坐标轴字体大小
        plt.yticks(fontname='Times New Roman', size=15)
        plt.xticks(fontname='Times New Roman', size=15)
        # x轴、y轴名称及标题
        plt.xlabel('frame', {'family': 'Times New Roman', 'size': 15})
        plt.ylabel('d/pixel', {'family': 'Times New Roman', 'size': 15})
        plt.ylim(0, 14)

        # 存放每个pattern的结果
        Result_pattern = {}
        c_ids = 0  # 颜色索引
        for ids1,p1 in enumerate(args.boxSize): # 跟踪框大小------6个
            for ids2,p2 in enumerate(args.upDatePattern): #模板更新阈值-------4个
                Time_list = []
                Distance_pointList = []  # 元素为list,[[1,2,3,...],[4,5,6,...],...],共5个list,每两个点之间的距离，用于可视化

                dirs_new = dirs_init.copy()
                labels_new = labels_init.copy()

                target_point = labels_new[0]  # 第一帧需要跟踪的点
                first_frame = cv2.imread(args.images + '/{}'.format(dirs_new[0]))  # 第一帧图像
                del (dirs_new[0])  # 剩余的图像,不包括firstFrame
                del (labels_new[0])  # 剩余的labels,不包括target_point
                print("dirs_new 长度= ", len(dirs_new))
                print('labels_new 长度 = ', len(labels_new))

                first_frame_copyed = first_frame.copy()
                first_frame_show = first_frame.copy()

                # 每次的帧及跟踪的点，存入字典中，便于后期的模板更新，key=当前帧数
                upDateDict = {} #{j：[frame,[target_point_dst[0], target_point_dst[1]],rect_x,rect_y,rect_w,rect_h],...}
                tracked_points = [] #存放每帧的预测的跟踪点 [[1,2],[3,4],[5,6],...]
                usedTime = []
                labels_last = []
                #连续帧
                j = 0 #通过j确保在模板更新时，不遗漏每一帧
                k = 1 #用于从upDateDict中，从后往前取保存的模板信息（图像+跟踪的点）
                #间隔帧
                # j = 0  # 通过j确保在模板更新时，不遗漏每一帧
                # k = 10  # 用于从upDateDict中，从后往前取保存的模板信息（图像+跟踪的点）
                action = False
                while j<len(dirs_new):
                    print('第{}张图'.format(j))
                    startTime = time.time()
                    # 连续读取后续所有图片
                    frame = cv2.imread(args.images + '/{}'.format(dirs_new[j]))
                    frame_copyed = frame.copy()
                    frame_show = frame.copy()

                    if (p1==100 or p1==150) and p2==10:
                        action = True
                        print('跟踪框={},模板更新 = {}'.format(p1,p2))
                        # 第一帧矩形框位置
                        rect_x = int(target_point[0] - int(p1 / 2))
                        rect_y = int(target_point[1] - int(p1 / 2))
                        rect_w = rect_h = p1

                        # 跟踪器初始化
                        init_state = [rect_x, rect_y, rect_w, rect_h]
                        trk.init(first_frame_copyed, init_state)

                        # 更新跟踪的矩形框
                        pos = trk.update(frame_copyed)
                        pos = _x1y1wh_to_xyxy(pos)
                        pos_int = [int(p) for p in pos]

                        # 第一帧掩膜
                        mask_first_frame = np.zeros_like(first_frame_copyed)
                        mask_first_frame[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w] = 255
                        first_frame_mask = cv2.bitwise_and(first_frame_copyed, mask_first_frame)

                        # 后续帧掩膜
                        mask_dst = np.zeros_like(frame_copyed)
                        mask_dst[pos_int[1]: pos_int[3], pos_int[0]: pos_int[2]] = 255
                        frame_mask = cv2.bitwise_and(frame_copyed, mask_dst)

                        # 特征检测
                        kp1, des1 = sift.detectAndCompute(first_frame_mask, None)
                        kp2, des2 = sift.detectAndCompute(frame_mask, None)
                        matches = flann.knnMatch(des2, des1, k=2)

                        # 特征匹配
                        good_matches = []
                        for match in matches:
                            if len(match) == 2:
                                if match[0].distance < 0.70 * match[1].distance:  # 小于0.7, 则认为特征点匹配成功
                                    good_matches.append(match[0])
                        #进行模板更新
                        if len(good_matches) < p2:
                            first_frame_copyed = upDateDict[j - k][0]
                            first_frame_show = upDateDict[j - k][0]
                            target_point = upDateDict[j - k][1]

                            print('*******good_matches个数 = {}<{}，模板更新*******'.format(len(good_matches), p2))
                            print("取upDateDict中第{}个".format(j - k))
                            k += 1
                            # k += 10
                            continue

                        # 求单应性矩阵H
                        src_pts_last = np.float32([kp1[good.trainIdx].pt for good in good_matches]).reshape(-1, 1, 2)
                        dst_pts_last = np.float32([kp2[good.queryIdx].pt for good in good_matches]).reshape(-1, 1, 2)
                        H, mask = cv2.findHomography(src_pts_last, dst_pts_last, cv2.RANSAC, 5.0)
                        if not isinstance(H, np.ndarray):  # 判断是否为空
                            first_frame_copyed = upDateDict[j - k][0]
                            first_frame_show = upDateDict[j - k][0]
                            target_point = upDateDict[j - k][1]
                            print('矩阵 H 为空,继续下一张图片')
                            print("H = ", H)
                            print("取upDateDict中第{}个".format(j - k))
                            k += 1
                            # k += 10
                            continue

                        # 目标点进行矩阵变换
                        target_point_dst = cv2.perspectiveTransform(np.array([[target_point]], dtype=np.float32), H)
                        target_point_dst = target_point_dst.reshape(-1)
                        print('=============结束=================')

                        # 画初始帧点
                        # cv2.circle(first_frame_show, (int(target_point[0]), int(target_point[1])), 10, (0, 0, 255), -1)
                        # 后续帧画点
                        cv2.circle(frame_show, (int(target_point_dst[0]), int(target_point_dst[1])), 10,(255, 0, 0), -1)  # 显示追踪到的目标点BGR
                        cv2.circle(frame_show, (int(labels_new[j][0]), int(labels_new[j][1])), 10, (0, 0, 255),-1)  # 画label
                        # 画框
                        # cv2.rectangle(first_frame_show, (rect_x - 2, rect_y - 2),(rect_x + rect_w + 2, rect_y + rect_h + 2), (0, 0, 255), 5)  # 画跟踪框
                        cv2.rectangle(frame_show, (pos_int[0] - 2, pos_int[1] - 2),(pos_int[2] + 2, pos_int[3] + 2), (255, 0, 0), 5)  # 画跟踪框，红色
                        cv2.namedWindow("show", cv2.WND_PROP_ASPECT_RATIO)
                        cv2.imshow('show', frame_show)
                        cv2.waitKey(1)

                        usedTime.append(time.time() - startTime)  # 每一帧所花的时间
                        tracked_points.append(target_point_dst)
                        labels_last.append(labels_new[j])
                        # 连续帧
                        j += 1
                        k = 1
                        # 间隔帧
                        # j += 10
                        # k = 10
                    if (p1==200 or p1==250) and p2==15:
                        action = True
                        print('跟踪框={},模板更新 = {}'.format(p1,p2))
                        # 第一帧矩形框位置
                        rect_x = int(target_point[0] - int(p1 / 2))
                        rect_y = int(target_point[1] - int(p1 / 2))
                        rect_w = rect_h = p1

                        # 跟踪器初始化
                        init_state = [rect_x, rect_y, rect_w, rect_h]
                        trk.init(first_frame_copyed, init_state)

                        # 更新跟踪的矩形框
                        pos = trk.update(frame_copyed)
                        pos = _x1y1wh_to_xyxy(pos)
                        pos_int = [int(p) for p in pos]

                        # 第一帧掩膜
                        mask_first_frame = np.zeros_like(first_frame_copyed)
                        mask_first_frame[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w] = 255
                        first_frame_mask = cv2.bitwise_and(first_frame_copyed, mask_first_frame)

                        # 后续帧掩膜
                        mask_dst = np.zeros_like(frame_copyed)
                        mask_dst[pos_int[1]: pos_int[3], pos_int[0]: pos_int[2]] = 255
                        frame_mask = cv2.bitwise_and(frame_copyed, mask_dst)

                        # 特征检测
                        kp1, des1 = sift.detectAndCompute(first_frame_mask, None)
                        kp2, des2 = sift.detectAndCompute(frame_mask, None)
                        matches = flann.knnMatch(des2, des1, k=2)

                        # 特征匹配
                        good_matches = []
                        for match in matches:
                            if len(match) == 2:
                                if match[0].distance < 0.70 * match[1].distance:  # 小于0.7, 则认为特征点匹配成功
                                    good_matches.append(match[0])
                        # 进行模板更新
                        if len(good_matches) < p2:
                            first_frame_copyed = upDateDict[j - k][0]
                            first_frame_show = upDateDict[j - k][0]
                            target_point = upDateDict[j - k][1]

                            print('*******good_matches个数 = {}<{}，模板更新*******'.format(len(good_matches), p2))
                            print("取upDateDict中第{}个".format(j - k))
                            k += 1
                            # k += 10
                            continue

                        # 求单应性矩阵H
                        src_pts_last = np.float32([kp1[good.trainIdx].pt for good in good_matches]).reshape(-1, 1,
                                                                                                            2)
                        dst_pts_last = np.float32([kp2[good.queryIdx].pt for good in good_matches]).reshape(-1, 1,
                                                                                                            2)
                        H, mask = cv2.findHomography(src_pts_last, dst_pts_last, cv2.RANSAC, 5.0)
                        if not isinstance(H, np.ndarray):  # 判断是否为空
                            first_frame_copyed = upDateDict[j - k][0]
                            first_frame_show = upDateDict[j - k][0]
                            target_point = upDateDict[j - k][1]
                            print('矩阵 H 为空,继续下一张图片')
                            print("H = ", H)
                            print("取upDateDict中第{}个".format(j - k))
                            k += 1
                            # k += 10
                            continue

                        # 目标点进行矩阵变换
                        target_point_dst = cv2.perspectiveTransform(np.array([[target_point]], dtype=np.float32), H)
                        target_point_dst = target_point_dst.reshape(-1)
                        print('=============结束=================')

                        # 画初始帧点
                        # cv2.circle(first_frame_show, (int(target_point[0]), int(target_point[1])), 10, (0, 0, 255), -1)
                        # 后续帧画点
                        cv2.circle(frame_show, (int(target_point_dst[0]), int(target_point_dst[1])), 10,
                                   (255, 0, 0), -1)  # 显示追踪到的目标点BGR
                        cv2.circle(frame_show, (int(labels_new[j][0]), int(labels_new[j][1])), 10, (0, 0, 255),
                                   -1)  # 画label
                        # 画框
                        # cv2.rectangle(first_frame_show, (rect_x - 2, rect_y - 2),(rect_x + rect_w + 2, rect_y + rect_h + 2), (0, 0, 255), 5)  # 画跟踪框
                        cv2.rectangle(frame_show, (pos_int[0] - 2, pos_int[1] - 2),
                                      (pos_int[2] + 2, pos_int[3] + 2), (255, 0, 0), 5)  # 画跟踪框，红色
                        cv2.namedWindow("show", cv2.WND_PROP_ASPECT_RATIO)
                        cv2.imshow('show', frame_show)
                        cv2.waitKey(1)

                        usedTime.append(time.time() - startTime)  # 每一帧所花的时间
                        tracked_points.append(target_point_dst)
                        labels_last.append(labels_new[j])
                        # 连续帧
                        j += 1
                        k = 1
                        # 间隔帧
                        # j += 10
                        # k = 10
                    # if p1==200 and p2=='No_update':
                    #     action = True
                    #     print('跟踪框={},模板更新 = {}'.format(p1,p2))
                    #     # 第一帧矩形框位置
                    #     rect_x = int(target_point[0] - int(p1 / 2))
                    #     rect_y = int(target_point[1] - int(p1 / 2))
                    #     rect_w = rect_h = p1
                    #
                    #     # 跟踪器初始化
                    #     init_state = [rect_x, rect_y, rect_w, rect_h]
                    #     trk.init(first_frame_copyed, init_state)
                    #
                    #     # 更新跟踪的矩形框
                    #     pos = trk.update(frame_copyed)
                    #     pos = _x1y1wh_to_xyxy(pos)
                    #     pos_int = [int(p) for p in pos]
                    #
                    #     # 第一帧掩膜
                    #     mask_first_frame = np.zeros_like(first_frame_copyed)
                    #     mask_first_frame[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w] = 255
                    #     first_frame_mask = cv2.bitwise_and(first_frame_copyed, mask_first_frame)
                    #
                    #     # 后续帧掩膜
                    #     mask_dst = np.zeros_like(frame_copyed)
                    #     mask_dst[pos_int[1]: pos_int[3], pos_int[0]: pos_int[2]] = 255
                    #     frame_mask = cv2.bitwise_and(frame_copyed, mask_dst)
                    #
                    #     # 特征检测
                    #     kp1, des1 = sift.detectAndCompute(first_frame_mask, None)
                    #     kp2, des2 = sift.detectAndCompute(frame_mask, None)
                    #     matches = flann.knnMatch(des2, des1, k=2)
                    #
                    #     # 特征匹配
                    #     good_matches = []
                    #     for match in matches:
                    #         if len(match) == 2:
                    #             if match[0].distance < 0.70 * match[1].distance:  # 小于0.7, 则认为特征点匹配成功
                    #                 good_matches.append(match[0])
                    #     # 无进行模板更新
                    #     # 求单应性矩阵H
                    #     src_pts_last = np.float32([kp1[good.trainIdx].pt for good in good_matches]).reshape(-1, 1,
                    #                                                                                         2)
                    #     dst_pts_last = np.float32([kp2[good.queryIdx].pt for good in good_matches]).reshape(-1, 1,
                    #                                                                                         2)
                    #     H, mask = cv2.findHomography(src_pts_last, dst_pts_last, cv2.RANSAC, 5.0)
                    #     if not isinstance(H, np.ndarray):  # 判断是否为空
                    #         first_frame_copyed = upDateDict[j - k][0]
                    #         first_frame_show = upDateDict[j - k][0]
                    #         target_point = upDateDict[j - k][1]
                    #         print('矩阵 H 为空,继续下一张图片')
                    #         print("H = ", H)
                    #         print("取upDateDict中第{}个".format(j - k))
                    #         # k += 1
                    #         k += 10
                    #         continue
                    #
                    #     # 目标点进行矩阵变换
                    #     target_point_dst = cv2.perspectiveTransform(np.array([[target_point]], dtype=np.float32), H)
                    #     target_point_dst = target_point_dst.reshape(-1)
                    #     print('=============结束=================')
                    #
                    #     # 画初始帧点
                    #     # cv2.circle(first_frame_show, (int(target_point[0]), int(target_point[1])), 10, (0, 0, 255), -1)
                    #     # 后续帧画点
                    #     cv2.circle(frame_show, (int(target_point_dst[0]), int(target_point_dst[1])), 10,
                    #                (255, 0, 0), -1)  # 显示追踪到的目标点BGR
                    #     cv2.circle(frame_show, (int(labels_new[j][0]), int(labels_new[j][1])), 10, (0, 0, 255),
                    #                -1)  # 画label
                    #     # 画框
                    #     # cv2.rectangle(first_frame_show, (rect_x - 2, rect_y - 2),(rect_x + rect_w + 2, rect_y + rect_h + 2), (0, 0, 255), 5)  # 画跟踪框
                    #     cv2.rectangle(frame_show, (pos_int[0] - 2, pos_int[1] - 2),
                    #                   (pos_int[2] + 2, pos_int[3] + 2), (255, 0, 0), 5)  # 画跟踪框，红色
                    #     cv2.namedWindow("show", cv2.WND_PROP_ASPECT_RATIO)
                    #     cv2.imshow('show', frame_show)
                    #     cv2.waitKey(1)
                    #
                    #     usedTime.append(time.time() - startTime)  # 每一帧所花的时间
                    #     tracked_points.append(target_point_dst)
                    #     labels_last.append(labels_new[j])
                    #     # 连续帧
                    #     # j += 1
                    #     # k = 1
                    #     # 间隔帧
                    #     j += 10
                    #     k = 10
                    if p1 == 300 and p2 == 5:
                        action = True
                        print('跟踪框={},模板更新 = {}'.format(p1,p2))
                        # 第一帧矩形框位置
                        rect_x = int(target_point[0] - int(p1 / 2))
                        rect_y = int(target_point[1] - int(p1 / 2))
                        rect_w = rect_h = p1

                        # 跟踪器初始化
                        init_state = [rect_x, rect_y, rect_w, rect_h]
                        trk.init(first_frame_copyed, init_state)

                        # 更新跟踪的矩形框
                        pos = trk.update(frame_copyed)
                        pos = _x1y1wh_to_xyxy(pos)
                        pos_int = [int(p) for p in pos]

                        # 第一帧掩膜
                        mask_first_frame = np.zeros_like(first_frame_copyed)
                        mask_first_frame[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w] = 255
                        first_frame_mask = cv2.bitwise_and(first_frame_copyed, mask_first_frame)

                        # 后续帧掩膜
                        mask_dst = np.zeros_like(frame_copyed)
                        mask_dst[pos_int[1]: pos_int[3], pos_int[0]: pos_int[2]] = 255
                        frame_mask = cv2.bitwise_and(frame_copyed, mask_dst)

                        # 特征检测
                        kp1, des1 = sift.detectAndCompute(first_frame_mask, None)
                        kp2, des2 = sift.detectAndCompute(frame_mask, None)
                        matches = flann.knnMatch(des2, des1, k=2)

                        # 特征匹配
                        good_matches = []
                        for match in matches:
                            if len(match) == 2:
                                if match[0].distance < 0.70 * match[1].distance:  # 小于0.7, 则认为特征点匹配成功
                                    good_matches.append(match[0])
                        # 进行模板更新
                        if len(good_matches) < p2:
                            first_frame_copyed = upDateDict[j - k][0]
                            first_frame_show = upDateDict[j - k][0]
                            target_point = upDateDict[j - k][1]

                            print('*******good_matches个数 = {}<{}，模板更新*******'.format(len(good_matches), p2))
                            print("取upDateDict中第{}个".format(j - k))
                            k += 1
                            # k += 10
                            continue

                        # 求单应性矩阵H
                        src_pts_last = np.float32([kp1[good.trainIdx].pt for good in good_matches]).reshape(-1, 1,
                                                                                                            2)
                        dst_pts_last = np.float32([kp2[good.queryIdx].pt for good in good_matches]).reshape(-1, 1,
                                                                                                            2)
                        H, mask = cv2.findHomography(src_pts_last, dst_pts_last, cv2.RANSAC, 5.0)
                        if not isinstance(H, np.ndarray):  # 判断是否为空
                            first_frame_copyed = upDateDict[j - k][0]
                            first_frame_show = upDateDict[j - k][0]
                            target_point = upDateDict[j - k][1]
                            print('矩阵 H 为空,继续下一张图片')
                            print("H = ", H)
                            print("取upDateDict中第{}个".format(j - k))
                            k += 1
                            # k += 10
                            continue

                        # 目标点进行矩阵变换
                        target_point_dst = cv2.perspectiveTransform(np.array([[target_point]], dtype=np.float32), H)
                        target_point_dst = target_point_dst.reshape(-1)
                        print('=============结束=================')

                        # 后续帧画点
                        cv2.circle(frame_show, (int(target_point_dst[0]), int(target_point_dst[1])), 10,(255, 0, 0), -1)  # 显示追踪到的目标点BGR
                        cv2.circle(frame_show, (int(labels_new[j][0]), int(labels_new[j][1])), 10, (0, 0, 255),-1)  # 画label
                        # 画框
                        cv2.rectangle(frame_show, (pos_int[0] - 2, pos_int[1] - 2),(pos_int[2] + 2, pos_int[3] + 2), (255, 0, 0), 5)  # 画跟踪框，红色
                        cv2.namedWindow("show", cv2.WND_PROP_ASPECT_RATIO)
                        cv2.imshow('show', frame_show)
                        cv2.waitKey(1)

                        usedTime.append(time.time() - startTime)  # 每一帧所花的时间
                        tracked_points.append(target_point_dst)
                        labels_last.append(labels_new[j])
                        # 连续帧
                        j += 1
                        k = 1
                        # 间隔帧
                        # j += 10
                        # k = 10
                    if p1=='No_Patch_tracking' and p2==15:
                        action = True
                        print('跟踪框={},模板更新 = {}'.format(p1,p2))
                        # 特征检测
                        kp1, des1 = sift.detectAndCompute(first_frame_copyed, None)
                        kp2, des2 = sift.detectAndCompute(frame_copyed, None)
                        matches = flann.knnMatch(des2, des1, k=2)

                        # 特征匹配
                        good_matches = []
                        for match in matches:
                            if len(match) == 2:
                                if match[0].distance < 0.70 * match[1].distance:  # 小于0.7, 则认为特征点匹配成功
                                    good_matches.append(match[0])
                        # 进行模板更新
                        if len(good_matches) < p2:
                            first_frame_copyed = upDateDict[j - k][0]
                            first_frame_show = upDateDict[j - k][0]
                            target_point = upDateDict[j - k][1]

                            print('*******good_matches个数 = {}<{}，模板更新*******'.format(len(good_matches), p2))
                            print("取upDateDict中第{}个".format(j - k))
                            k += 1
                            # k += 10
                            continue

                        # 求单应性矩阵H
                        src_pts_last = np.float32([kp1[good.trainIdx].pt for good in good_matches]).reshape(-1, 1,
                                                                                                            2)
                        dst_pts_last = np.float32([kp2[good.queryIdx].pt for good in good_matches]).reshape(-1, 1,
                                                                                                            2)
                        H, mask = cv2.findHomography(src_pts_last, dst_pts_last, cv2.RANSAC, 5.0)
                        if not isinstance(H, np.ndarray):  # 判断是否为空
                            first_frame_copyed = upDateDict[j - k][0]
                            first_frame_show = upDateDict[j - k][0]
                            target_point = upDateDict[j - k][1]
                            print('矩阵 H 为空,继续下一张图片')
                            print("H = ", H)
                            print("取upDateDict中第{}个".format(j - k))
                            k += 1
                            # k += 10
                            continue

                        # 目标点进行矩阵变换
                        target_point_dst = cv2.perspectiveTransform(np.array([[target_point]], dtype=np.float32), H)
                        target_point_dst = target_point_dst.reshape(-1)
                        print('=============结束=================')

                        # 画初始帧点
                        # cv2.circle(first_frame_show, (int(target_point[0]), int(target_point[1])), 10, (0, 0, 255), -1)
                        # 后续帧画点
                        cv2.circle(frame_show, (int(target_point_dst[0]), int(target_point_dst[1])), 10,(255, 0, 0), -1)  # 显示追踪到的目标点BGR
                        cv2.circle(frame_show, (int(labels_new[j][0]), int(labels_new[j][1])), 10, (0, 0, 255),-1)  # 画label
                        # 画框
                        # cv2.rectangle(first_frame_show, (rect_x - 2, rect_y - 2),(rect_x + rect_w + 2, rect_y + rect_h + 2), (0, 0, 255), 5)  # 画跟踪框
                        cv2.rectangle(frame_show, (pos_int[0] - 2, pos_int[1] - 2),(pos_int[2] + 2, pos_int[3] + 2), (255, 0, 0), 5)  # 画跟踪框，红色
                        cv2.namedWindow("show", cv2.WND_PROP_ASPECT_RATIO)
                        cv2.imshow('show', frame_show)
                        cv2.waitKey(1)

                        usedTime.append(time.time() - startTime)  # 每一帧所花的时间
                        tracked_points.append(target_point_dst)
                        labels_last.append(labels_new[j])
                        # 连续帧
                        j += 1
                        k = 1
                        # 间隔帧
                        # j += 10
                        # k = 10
                    if not action:
                        action = False
                        j += 1
                        k = 1

                        # j += 10
                        # k = 10
                        print('p1 = {},p2 = {}:跳过不执行'.format(p1,p2),)
                        print("====================下一轮=================")
                        continue
                    if action:
                        ###################模板更新###################
                        # 将当前的图像帧、跟踪的点，矩形框位置存到字典中，key = 当前帧数j,value = [当前帧，跟踪的点位置，矩形框位置]
                        upDateDict[j] = [frame_copyed, [target_point_dst[0], target_point_dst[1]]]
                        #连续帧
                        k = 1
                        # 间隔帧
                        # k = 10
                if action:
                    # 评估每次处理完所有照片后的结果，并存入list中
                    print('******************')
                    print("tracked_points 长度 ={},labels_last长度 ={} ".format(len(tracked_points),len(labels_last)))
                    # print("查看点的数量，按空格键继续")
                    # cv2.waitKey(0)

                    AR5, AR6, AR7 = AR_evaluate(tracked_points, labels_last)
                    MSE = MSE_evaluate(tracked_points, labels_last)
                    Distance = Euclidean_evaluate(tracked_points, labels_last)
                    Time_list=mean(usedTime)
                    Result_pattern['{}_sift_{}_continue'.format(p1,p2)] = ['AR5 = {},AR6 = {},AR7 = {},Distance = {},MSE = {},Time = {}'.format(AR5, AR6, AR7,Distance, MSE,Time_list)]

                    ##########用于获取距离可视化的距离数据##########
                    # 每两点之间的距离,用于可视化
                    dis = [np.sqrt(np.power(tracked_points[i][0] - labels_last[i][0], 2) + np.power(tracked_points[i][1] - labels_last[i][1], 2)) for i in range(len(tracked_points))]
                    Distance_pointList=dis
                    print('Distance_pointList 长度 = ',len(Distance_pointList))
                    # print("查看可视化点的数量，按空格键继续")
                    # print('======================================')
                    # cv2.waitKey(0)

                    # 画预测点与真实点距离
                    lineLegendName = str(p1)+'_'+'sift'+'_'+str(p2)
                    lineLegendColor = args.LegendColor[c_ids]
                    c_ids+=1
                    plt.plot(Distance_pointList, color=lineLegendColor, linewidth=2, label=lineLegendName)
                    plt.legend(prop={"family": "Times New Roman", "size": 15}, loc='upper left')

                # cv2.destroyWindow('show')
        #存图像
        plt.savefig('./实验结果/5.9/last_interval.jpg')
        plt.close('all')

        print('Result_pattern = ', Result_pattern)
        # 保存字典，便于统计
        file = open('./实验结果/5.9/last_interval.txt', 'a')
        file.write(str(Result_pattern)+'\n')
        file.close()
        break

if __name__ == "__main__":
    args = parse_args()
    main(args)
