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
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致

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
    # 特征检测模式
    parser.add_argument('--kpDetect', dest='kpDetect',default=['sift', 'surf', 'No_matching'])  # 特征匹配求单应性矩阵; 无特征匹配，直接位置映射
    parser.add_argument('--kpLegendName', dest='kpLegendName', default=['sift', 'surf', 'No_matching'],help='kpLegend title')
    # 有无模板更新及阈值
    parser.add_argument('--upDatePattern', dest='upDatePattern', default=['No_update', 5,6,7])
    parser.add_argument('--upDateLegendName', dest='upDateLegendName', default=['No_update', '5', '6', '7'], help='upDatePattern title')

    parser.add_argument('--LegendColor',dest = 'LegendColor',default=['green','blue','red','purple','brown','darkgoldenrod'],help ='LegendColor')
    # parser.add_argument('--LegendMarker', dest='LegendMarker',default=['s', 'x', '+', '*'],help='LegendMarker')
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
    for i in range(len(tracked_points)):
        dis = np.sqrt(np.power(labelsNew[i][0]-tracked_points[i][0],2)+np.power(labelsNew[i][1]-tracked_points[i][1],2))
        if dis <=2:
            pos5+=1
        if dis <=3:
            pos6+=1
        if dis <=4:
            pos7+=1
    AR5 = pos5/len(tracked_points)
    AR6 = pos6/ len(tracked_points)
    AR7 = pos7/ len(tracked_points)

    return AR5,AR6,AR7

def MSE_evaluate(tracked_points, labelsNew): # MSE评估
    MSE_result = []
    for i in range(len(tracked_points)):
        mse_dis = np.power(labelsNew[i][0] - tracked_points[i][0], 2) + np.power(labelsNew[i][1] - tracked_points[i][1],2)
        MSE_result.append(mse_dis)
    MSE = np.sum(MSE_result) / len(tracked_points)
    return MSE

def Euclidean_evaluate(tracked_points, labelsNew): # 欧氏距离评估
    Euclidean_list = []
    for i in range(len(tracked_points)):
        dis = np.sqrt(np.power(labelsNew[i][0]-tracked_points[i][0],2)+np.power(labelsNew[i][1]-tracked_points[i][1],2))
        Euclidean_list.append(dis)
    Euclidean= np.sum(Euclidean_list)/len(tracked_points)
    return Euclidean


def main(args):
    # 所有图片
    dirs_init = os.listdir(args.images)  # dirs =  ['1.jpg', '10.jpg', '100.jpg', '101.jpg',...,]
    dirs_init.sort(key=lambda x: int(x[:-4]))  # dirs =  ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg',...,]
    # dirs_init=dirs_init[:150]
    # 所有标签
    labels_init = get_groundTruth(args.labels)
    # labels_init=labels_init[:150]
    # 跟踪器初始化
    trk = TrackerSiamFC(net_path=args.model)
    ## 初始化特征检测器
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()

    # 特征匹配初始化
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # KD树个数：5
    search_params = dict(checks=50)  # 遍历次数
    flann = cv2.FlannBasedMatcher(index_params, search_params)  # 创建匹配器

    while(True):
        # 存放每个pattern的结果
        Result_pattern = {}
        for ids1,p1 in enumerate(args.boxSize): # 跟踪框大小------6个
            for ids2,p2 in enumerate(args.kpDetect): # 特征检测模式------3个
                # 画布
                plt.figure(dpi=900)
                # 修改坐标轴字体大小
                plt.yticks(fontname='Times New Roman', size=15)
                plt.xticks(fontname='Times New Roman', size=15)
                # x轴、y轴名称及标题
                plt.xlabel('frame', {'family': 'Times New Roman', 'size': 15})
                plt.ylabel('d/pixel', {'family': 'Times New Roman', 'size': 15})
                for ids3,p3 in enumerate(args.upDatePattern): #模板更新阈值-------4个
                    AR5_list = []  # 元素为标量
                    AR6_list = []
                    AR7_list = []
                    Distance_list = []
                    MSE_list = []
                    Time_list = []
                    Distance_pointList = []  # 元素为list,[[1,2,3,...],[4,5,6,...],...],共5个list,每两个点之间的距离，用于可视化
                    for firstFrameNUM in range(1): # 每次循环都有一个AR，MSE, Distance和Time
                        dirs_new = dirs_init.copy()
                        labels_new = labels_init.copy()
                        print("=======================起始=======================")
                        # print("pattern = ", p)
                        # print("起始帧为第{}帧".format(firstFrameNUM))
                        print("dirs_init 长度= ", len(dirs_init))
                        print('labels_init 长度 = ', len(labels_init))

                        target_point = labels_new[firstFrameNUM]  # 第一帧需要跟踪的点
                        first_frame = cv2.imread(args.images + '/{}'.format(dirs_new[firstFrameNUM]))  # 第一帧图像
                        first_frame_copyed = first_frame.copy()
                        first_frame_show = first_frame.copy()

                        del (dirs_new[0:firstFrameNUM+1])  # 剩余的图像,不包括firstFrame
                        del (labels_new[0:firstFrameNUM+1])  # 剩余的labels,不包括target_point
                        print("dirs_new 长度= ",len(dirs_new))
                        print('labels_new 长度 = ',len(labels_new))

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
                        while j<len(dirs_new):
                            print('第{}张图'.format(j))
                            startTime = time.time()
                            # print("跟踪框：{},特征检测：{},模板更新:{} ".format(p1,p2,p3))
                            # print("起始帧为第{}帧".format(firstFrameNUM))
                            # print("第{}张图片".format(j))

                            # 连续读取后续所有图片
                            frame = cv2.imread(args.images + '/{}'.format(dirs_new[j]))
                            frame_copyed = frame.copy()
                            frame_show = frame.copy()

                            # p1有跟踪框
                            if isinstance(p1,int):
                                print('跟踪框：',p1)
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

                                # 有特征检测
                                if p2=='sift':
                                    print('特征检测：',p2)
                                    kp1, des1 = sift.detectAndCompute(first_frame_mask, None)
                                    kp2, des2 = sift.detectAndCompute(frame_mask, None)
                                    matches = flann.knnMatch(des2, des1, k=2)
                                    #有无模板更新
                                elif p2=='surf':
                                    print('特征检测：', p2)
                                    kp1, des1 = surf.detectAndCompute(first_frame_mask, None)
                                    kp2, des2 = surf.detectAndCompute(frame_mask, None)
                                    matches = flann.knnMatch(des2, des1, k=2)
                                    # 有无模板更新
                                #无特征检测
                                else:
                                    print('特征检测：无特征检测，直接位置映射')
                                    target_point_dst = [(pos_int[0] + pos_int[2]) / 2, (pos_int[1] + pos_int[3]) / 2]
                                    print('=============结束=================')

                                # 特征匹配
                                if p2 == "sift" or p2 == 'surf':
                                    print('特征匹配：',p2)
                                    good_matches = []
                                    for match in matches:
                                        if len(match) == 2:
                                            if match[0].distance < 0.60 * match[1].distance:  # 小于0.6, 则认为特征点匹配成功
                                                good_matches.append(match[0])
                                    q=len(good_matches)
                                    print('匹配点个数',q)
                                    #进行模板更新
                                    if isinstance(p3,int) and len(good_matches) < p3:
                                        first_frame_copyed = upDateDict[j - k][0]
                                        first_frame_show = upDateDict[j - k][0]
                                        target_point = upDateDict[j - k][1]

                                        print('*******good_matches个数 = {}<{}，模板更新*******'.format(len(good_matches), p3))
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

                                usedTime.append(time.time() - startTime)  # 每一帧所花的时间
                                tracked_points.append(target_point_dst)#
                                labels_last.append(labels_new[j])


                            # p1无跟踪框
                            else:
                                print('无跟踪框')
                                # 特征检测
                                if p2=='sift':
                                    print('特征检测：',p2)
                                    kp1, des1 = sift.detectAndCompute(first_frame_copyed, None)
                                    kp2, des2 = sift.detectAndCompute(frame_copyed, None)
                                    matches = flann.knnMatch(des2, des1, k=2)
                                elif p2=='surf':
                                    print('特征检测：', p2)
                                    kp1, des1 = surf.detectAndCompute(first_frame_copyed, None)
                                    kp2, des2 = surf.detectAndCompute(frame_copyed, None)
                                    matches = flann.knnMatch(des2, des1, k=2)
                                else:#p2 = No_matching,无特征检测，直接位置映射。无p3
                                    print('无特征检测，直接位置映射')
                                    x_ratio = target_point[0] / first_frame.shape[1]
                                    y_ratio = target_point[1] / first_frame.shape[0]
                                    target_point_dst = [frame.shape[1] * x_ratio, frame.shape[0] * y_ratio]
                                    print('=============结束=================')

                                # 特征匹配
                                if p2 == "sift" or p2 == 'surf':
                                    print('特征匹配：',p2)
                                    good_matches = []
                                    for match in matches:
                                        if len(match) == 2:
                                            if match[0].distance < 0.60 * match[1].distance:  # 小于0.6, 则认为特征点匹配成功
                                                good_matches.append(match[0])
                                            # print(len(good_matches))
                                    # 进行模板更新
                                    if isinstance(p3, int) and len(good_matches) < p3:
                                        first_frame_copyed = upDateDict[j - k][0]
                                        first_frame_show = upDateDict[j - k][0]
                                        target_point = upDateDict[j - k][1]

                                        print('*******good_matches个数 = {}<{}，模板更新*******'.format(
                                            len(good_matches), p3))
                                        print("取upDateDict中第{}个".format(j - k))
                                        k += 1
                                        # k += 10
                                        continue

                                    # 求单应性矩阵H
                                    src_pts_last = np.float32([kp1[good.trainIdx].pt for good in good_matches]).reshape(-1, 1, 2)
                                    dst_pts_last = np.float32([kp2[good.queryIdx].pt for good in good_matches]).reshape(-1, 1, 2)
                                    H, mask = cv2.findHomography(src_pts_last, dst_pts_last, cv2.RANSAC,5.0)
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

                                usedTime.append(time.time() - startTime)  # 每一帧所花的时间
                                tracked_points.append(target_point_dst)
                                labels_last.append(labels_new[j])

                            ##############################每帧图像显示############################
                            # 初始帧画点
                            cv2.circle(first_frame_show, (int(target_point[0]), int(target_point[1])), 10, (0, 0, 255), -1)
                            #后续帧画点
                            cv2.circle(frame_show, (int(target_point_dst[0]), int(target_point_dst[1])), 10, (255, 0, 0), -1)  # 显示追踪到的目标点BGR
                            cv2.circle(frame_show, (int(labels_new[j][0]), int(labels_new[j][1])), 10, (0, 0, 255),-1)  # 画label
                            #初始帧、后续帧画矩形框
                            if isinstance(p1,int):
                                cv2.rectangle(first_frame_show, (rect_x - 2, rect_y - 2),(rect_x + rect_w + 2, rect_y + rect_h + 2), (0, 0, 255), 5)  # 画跟踪框
                                cv2.rectangle(frame_show, (pos_int[0] - 2, pos_int[1] - 2), (pos_int[2] + 2, pos_int[3] + 2),(255,0, 0), 5)  # 画跟踪框，红色
                            # 如果匹配
                            if p2 != 'No_matching':
                                matchImage = cv2.drawMatches(frame_show, kp2,first_frame_show, kp1,  good_matches,None,flags = 2)
                                cv2.namedWindow("show", cv2.WND_PROP_ASPECT_RATIO)
                                cv2.imshow('show', matchImage)
                                cv2.waitKey(1)
                            #如果不匹配
                            else:
                                hstackImage = np.hstack([first_frame_show,frame_show])
                                cv2.namedWindow("show",cv2.WND_PROP_ASPECT_RATIO)
                                cv2.imshow('show',hstackImage)
                                cv2.waitKey(1)

                            ###################模板更新###################
                            # 将当前的图像帧、跟踪的点，矩形框位置存到字典中，key = 当前帧数j,value = [当前帧，跟踪的点位置，矩形框位置]
                            upDateDict[j] = [frame_copyed, [target_point_dst[0], target_point_dst[1]]]#跟踪点

                            #连续帧
                            j += 1
                            k = 1
                            # 间隔帧
                            # j += 10
                            # k = 10

                        # 评估每次处理完所有照片后的结果，并存入list中
                        print('******************')
                        print('firstFrameNUM = {} 时 '.format(firstFrameNUM))
                        print("tracked_points 长度 =", len(tracked_points))
                        print("labels_last 长度 =", len(labels_last))
                        # print("按空格键继续")
                        # cv2.waitKey(0)
                        tracked_points_=np.array(tracked_points)
                        track_num = len(tracked_points_)
                        print(track_num)
                        goodmatch_path = '实验结果5_6_7'
                        Name = args.boxLegendName[ids1] + '_' + args.kpLegendName[ids2] + '_' + args.upDateLegendName[ids3]
                        if not os.path.exists(goodmatch_path):
                            os.makedirs(goodmatch_path)
                        np.savetxt(os.path.join(goodmatch_path, f'{Name}.txt'), tracked_points_)

                        AR5, AR6, AR7 = AR_evaluate(tracked_points, labels_last)
                        MSE = MSE_evaluate(tracked_points, labels_last)
                        Distance = Euclidean_evaluate(tracked_points, labels_last)

                        AR5_list.append(AR5)
                        AR6_list.append(AR6)
                        AR7_list.append(AR7)
                        MSE_list.append(MSE)
                        Distance_list.append(Distance)
                        Time_list.append(mean(usedTime))
                        # 每两点之间的距离,用于可视化
                        dis = [np.sqrt(np.power(tracked_points[i][0] - labels_last[i][0], 2) + np.power(tracked_points[i][1] - labels_last[i][1], 2)) for i in range(len(tracked_points))]
                        Distance_pointList.append(dis)

                    # 每个pattern
                    print('******************')
                    print("跟踪框：{},特征检测：{},模板更新:{}时 ".format(p1, p2, p3))
                    print('AR5_list 长度(应该为5) = ', len(AR5_list))
                    print('AR6_list 长度(应该为5) = ', len(AR6_list))
                    print('AR7_list 长度(应该为5) = ', len(AR7_list))
                    print('MSE_list 长度(应该为5) = ', len(MSE_list))
                    print('Distance_list 长度(应该为5) = ', len(Distance_list))
                    print('Time_list 长度(应该为5) = ', len(Time_list))
                    print('画点的Distance_pointList 长度(应该为5) = ', len(Distance_pointList))
                    AR5_mean = mean(AR5_list)
                    AR6_mean = mean(AR6_list)
                    AR7_mean = mean(AR7_list)
                    MSE_mean = mean(MSE_list)
                    Distance_mean = mean(Distance_list)
                    Time_mean = mean(Time_list)

                    # Distance_pointList中选择最短的一个list,将其他list从前往后截取，与list一样长度，再求平均
                    minList = min(Distance_pointList, key=len)
                    minDis = len(minList)
                    for i in range(len(Distance_pointList)):  # 循环结束后，Distance_pointList中5个list长度一样
                        delta = abs(len(Distance_pointList[i]) - minDis)
                        del (Distance_pointList[i][0:delta])
                    Distance_point = np.mean(Distance_pointList, axis=0).tolist()  # 用于可视化，画图
                    print("画图的点的个数：{} = 标签个数：{}".format(len(Distance_point), minDis))
                    print("====================结束=======================")
                    # print('按空格键继续')
                    # print("=============================================")
                    # cv2.waitKey(0)

                    Result_pattern['{}_{}_{}continue'.format(p1,p2,p3)] = ['AR5 = {},AR6 = {},AR7 = {},Distance = {},MSE = {},Time = {}'.format(AR5_mean, AR6_mean, AR7_mean,Distance_mean, MSE_mean,Time_mean)]

                    ##########用于获取距离可视化的距离数据##########
                    # 画预测点与真实点距离
                    lineLegendName = args.boxLegendName[ids1]+'_'+args.kpLegendName[ids2]+'_'+args.upDateLegendName[ids3]
                    lineLegendColor = args.LegendColor[ids3]
                    # lineLegendStyle = args.LegendStyle[ids2]
                    # lineLegendMarker = args.LegendMarker[ids3]
                    plt.plot(Distance_point, color=lineLegendColor, linewidth=2, label=lineLegendName)
                    plt.legend(prop={"family": "Times New Roman", "size": 15}, loc='upper left')

                plt.savefig('./实验结果5_6_7/all_{}_{}.jpg'.format(p1,p2))
                plt.close('all')

        print('Result_pattern = ', Result_pattern)
        # 保存字典，便于统计
        file = open('./实验结果5_6_7/all.txt', 'a')
        file.write(str(Result_pattern)+'\n')
        file.close()
        break

if __name__ == "__main__":
    args = parse_args()
    main(args)
