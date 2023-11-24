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
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_args():
    """
    args for testing.
    """
    parser = argparse.ArgumentParser(description='PyTorch SiamFC Tracking Demo')
    parser.add_argument('--model', dest='model', default='./checkPoint/siamfc/model.pth', help='pretrained model')
    parser.add_argument('--images', dest='images', default='./Point_Annotations/images/video1', help='video path')
    parser.add_argument('--labels',dest = 'labels', default='./Point_Annotations/csv/video1_point.csv', help='label path')

    parser.add_argument('--boxSize',dest = 'boxSize',default=[100,150,200,250,300,'No_Patch_tracking'])
    parser.add_argument('--boxLegendName',dest = 'boxLegendName',default=['s=100','s=150','s=200','s=250','s=300','No_Patch_tracking'],help= 'boxLegend title')

    parser.add_argument('--kpDetect', dest='kpDetect',default=['sift', 'surf', 'No_matching'])
    parser.add_argument('--kpLegendName', dest='kpLegendName', default=['sift', 'surf', 'No_matching'],help='kpLegend title')

    parser.add_argument('--upDatePattern', dest='upDatePattern', default=['No_update', 5,6,7])
    parser.add_argument('--upDateLegendName', dest='upDateLegendName', default=['No_update', '5', '6', '7'], help='upDatePattern title')

    parser.add_argument('--LegendColor',dest = 'LegendColor',default=['green','blue','red','purple','brown','darkgoldenrod'],help ='LegendColor')
    parser.add_argument('--LegendStyle', dest='LegendStyle', default=['solid', 'dashed', 'dotted'],help='LegendStyle')
    args = parser.parse_args()

    return args

def _x1y1wh_to_xyxy(bbox_x1y1wh):
    x1, y1, w, h = bbox_x1y1wh
    x2 = x1+w
    y2 = y1+h
    return x1, y1, x2, y2


def get_groundTruth(path):
    with open(path,'r') as f:
        reader = csv.reader(f)
        next(reader)
        labels = [[float(i[0]),float(i[1])] for i in reader]
        return labels


def AR_evaluate(tracked_points, labelsNew):
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

def MSE_evaluate(tracked_points, labelsNew):
    MSE_result = []
    for i in range(len(tracked_points)):
        mse_dis = np.power(labelsNew[i][0] - tracked_points[i][0], 2) + np.power(labelsNew[i][1] - tracked_points[i][1],2)
        MSE_result.append(mse_dis)
    MSE = np.sum(MSE_result) / len(tracked_points)
    return MSE

def Euclidean_evaluate(tracked_points, labelsNew):
    Euclidean_list = []
    for i in range(len(tracked_points)):
        dis = np.sqrt(np.power(labelsNew[i][0]-tracked_points[i][0],2)+np.power(labelsNew[i][1]-tracked_points[i][1],2))
        Euclidean_list.append(dis)
    Euclidean= np.sum(Euclidean_list)/len(tracked_points)
    return Euclidean


def main(args):
    dirs_init = os.listdir(args.images)  # dirs =  ['1.jpg', '10.jpg', '100.jpg', '101.jpg',...,]
    dirs_init.sort(key=lambda x: int(x[:-4]))  # dirs =  ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg',...,]
    # dirs_init=dirs_init[:150]
    labels_init = get_groundTruth(args.labels)
    # labels_init=labels_init[:150]
    trk = TrackerSiamFC(net_path=args.model)
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    while(True):
        Result_pattern = {}
        for ids1,p1 in enumerate(args.boxSize):
            for ids2,p2 in enumerate(args.kpDetect):
                plt.figure(dpi=900)
                plt.yticks(fontname='Times New Roman', size=15)
                plt.xticks(fontname='Times New Roman', size=15)
                plt.xlabel('frame', {'family': 'Times New Roman', 'size': 15})
                plt.ylabel('d/pixel', {'family': 'Times New Roman', 'size': 15})
                for ids3,p3 in enumerate(args.upDatePattern):
                    AR5_list = []
                    AR6_list = []
                    AR7_list = []
                    Distance_list = []
                    MSE_list = []
                    Time_list = []
                    Distance_pointList = []
                    for firstFrameNUM in range(1):
                        dirs_new = dirs_init.copy()
                        labels_new = labels_init.copy()
                        print("==============================================")
                        # print("pattern = ", p)

                        # print("dirs_init len= ", len(dirs_init))
                        # print('labels_init len = ', len(labels_init))

                        target_point = labels_new[firstFrameNUM]
                        first_frame = cv2.imread(args.images + '/{}'.format(dirs_new[firstFrameNUM]))
                        first_frame_copyed = first_frame.copy()
                        first_frame_show = first_frame.copy()

                        del (dirs_new[0:firstFrameNUM+1])
                        del (labels_new[0:firstFrameNUM+1])
                        # print("dirs_new len= ",len(dirs_new))
                        # print('labels_new len = ',len(labels_new))

                        upDateDict = {} #{j：[frame,[target_point_dst[0], target_point_dst[1]],rect_x,rect_y,rect_w,rect_h],...}
                        tracked_points = []
                        usedTime = []
                        labels_last = []
                        j = 0
                        k = 1
                        while j<len(dirs_new):
                            # print('{}th image'.format(j))
                            startTime = time.time()
                            frame = cv2.imread(args.images + '/{}'.format(dirs_new[j]))
                            frame_copyed = frame.copy()
                            frame_show = frame.copy()

                            if isinstance(p1,int):
                                # print('box',p1)
                                rect_x = int(target_point[0] - int(p1 / 2))
                                rect_y = int(target_point[1] - int(p1 / 2))
                                rect_w = rect_h = p1

                                init_state = [rect_x, rect_y, rect_w, rect_h]
                                trk.init(first_frame_copyed, init_state)

                                pos = trk.update(frame_copyed)
                                pos = _x1y1wh_to_xyxy(pos)
                                pos_int = [int(p) for p in pos]

                                mask_first_frame = np.zeros_like(first_frame_copyed)
                                mask_first_frame[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w] = 255
                                first_frame_mask = cv2.bitwise_and(first_frame_copyed, mask_first_frame)

                                mask_dst = np.zeros_like(frame_copyed)
                                mask_dst[pos_int[1]: pos_int[3], pos_int[0]: pos_int[2]] = 255
                                frame_mask = cv2.bitwise_and(frame_copyed, mask_dst)

                                if p2=='sift':
                                    # print('frature',p2)
                                    kp1, des1 = sift.detectAndCompute(first_frame_mask, None)
                                    kp2, des2 = sift.detectAndCompute(frame_mask, None)
                                    matches = flann.knnMatch(des2, des1, k=2)

                                elif p2=='surf':
                                    kp1, des1 = surf.detectAndCompute(first_frame_mask, None)
                                    kp2, des2 = surf.detectAndCompute(frame_mask, None)
                                    matches = flann.knnMatch(des2, des1, k=2)
                                else:
                                    target_point_dst = [(pos_int[0] + pos_int[2]) / 2, (pos_int[1] + pos_int[3]) / 2]

                                if p2 == "sift" or p2 == 'surf':
                                    good_matches = []
                                    for match in matches:
                                        if len(match) == 2:
                                            if match[0].distance < 0.60 * match[1].distance:
                                                good_matches.append(match[0])
                                    q=len(good_matches)

                                    if isinstance(p3,int) and len(good_matches) < p3:
                                        first_frame_copyed = upDateDict[j - k][0]
                                        first_frame_show = upDateDict[j - k][0]
                                        target_point = upDateDict[j - k][1]
                                        k += 1
                                        continue


                                    src_pts_last = np.float32([kp1[good.trainIdx].pt for good in good_matches]).reshape(-1, 1, 2)
                                    dst_pts_last = np.float32([kp2[good.queryIdx].pt for good in good_matches]).reshape(-1, 1, 2)
                                    H, mask = cv2.findHomography(src_pts_last, dst_pts_last, cv2.RANSAC, 5.0)
                                    if not isinstance(H, np.ndarray):
                                        first_frame_copyed = upDateDict[j - k][0]
                                        first_frame_show = upDateDict[j - k][0]
                                        target_point = upDateDict[j - k][1]
                                        k += 1
                                        continue

                                    target_point_dst = cv2.perspectiveTransform(np.array([[target_point]], dtype=np.float32), H)
                                    target_point_dst = target_point_dst.reshape(-1)

                                usedTime.append(time.time() - startTime)
                                tracked_points.append(target_point_dst)
                                labels_last.append(labels_new[j])


                            else:
                                if p2=='sift':
                                    kp1, des1 = sift.detectAndCompute(first_frame_copyed, None)
                                    kp2, des2 = sift.detectAndCompute(frame_copyed, None)
                                    matches = flann.knnMatch(des2, des1, k=2)
                                elif p2=='surf':
                                    kp1, des1 = surf.detectAndCompute(first_frame_copyed, None)
                                    kp2, des2 = surf.detectAndCompute(frame_copyed, None)
                                    matches = flann.knnMatch(des2, des1, k=2)
                                else:
                                    x_ratio = target_point[0] / first_frame.shape[1]
                                    y_ratio = target_point[1] / first_frame.shape[0]
                                    target_point_dst = [frame.shape[1] * x_ratio, frame.shape[0] * y_ratio]

                                if p2 == "sift" or p2 == 'surf':
                                    good_matches = []
                                    for match in matches:
                                        if len(match) == 2:
                                            if match[0].distance < 0.60 * match[1].distance:
                                                good_matches.append(match[0])
                                            # print(len(good_matches))
                                    if isinstance(p3, int) and len(good_matches) < p3:
                                        first_frame_copyed = upDateDict[j - k][0]
                                        first_frame_show = upDateDict[j - k][0]
                                        target_point = upDateDict[j - k][1]
                                        k += 1
                                        # k += 10
                                        continue
                                    src_pts_last = np.float32([kp1[good.trainIdx].pt for good in good_matches]).reshape(-1, 1, 2)
                                    dst_pts_last = np.float32([kp2[good.queryIdx].pt for good in good_matches]).reshape(-1, 1, 2)
                                    H, mask = cv2.findHomography(src_pts_last, dst_pts_last, cv2.RANSAC,5.0)
                                    if not isinstance(H, np.ndarray):
                                        first_frame_copyed = upDateDict[j - k][0]
                                        first_frame_show = upDateDict[j - k][0]
                                        target_point = upDateDict[j - k][1]
                                        k += 1
                                        # k += 10
                                        continue
                                    target_point_dst = cv2.perspectiveTransform(np.array([[target_point]], dtype=np.float32), H)
                                    target_point_dst = target_point_dst.reshape(-1)

                                usedTime.append(time.time() - startTime)
                                tracked_points.append(target_point_dst)
                                labels_last.append(labels_new[j])


                            cv2.circle(first_frame_show, (int(target_point[0]), int(target_point[1])), 10, (0, 0, 255), -1)
                            cv2.circle(frame_show, (int(target_point_dst[0]), int(target_point_dst[1])), 10, (255, 0, 0), -1)
                            cv2.circle(frame_show, (int(labels_new[j][0]), int(labels_new[j][1])), 10, (0, 0, 255),-1)
                            if isinstance(p1,int):
                                cv2.rectangle(first_frame_show, (rect_x - 2, rect_y - 2),(rect_x + rect_w + 2, rect_y + rect_h + 2), (0, 0, 255), 5)
                                cv2.rectangle(frame_show, (pos_int[0] - 2, pos_int[1] - 2), (pos_int[2] + 2, pos_int[3] + 2),(255,0, 0), 5)

                            if p2 != 'No_matching':
                                matchImage = cv2.drawMatches(frame_show, kp2,first_frame_show, kp1,  good_matches,None,flags = 2)
                                cv2.namedWindow("show", cv2.WND_PROP_ASPECT_RATIO)
                                cv2.imshow('show', matchImage)
                                cv2.waitKey(1)

                            else:
                                hstackImage = np.hstack([first_frame_show,frame_show])
                                cv2.namedWindow("show",cv2.WND_PROP_ASPECT_RATIO)
                                cv2.imshow('show',hstackImage)
                                cv2.waitKey(1)

                            upDateDict[j] = [frame_copyed, [target_point_dst[0], target_point_dst[1]]]
                            j += 1
                            k = 1
                        tracked_points_=np.array(tracked_points)
                        track_num = len(tracked_points_)
                        print(track_num)
                        goodmatch_path = 'your path'
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
                        dis = [np.sqrt(np.power(tracked_points[i][0] - labels_last[i][0], 2) + np.power(tracked_points[i][1] - labels_last[i][1], 2)) for i in range(len(tracked_points))]
                        Distance_pointList.append(dis)
                    # AR5_mean = mean(AR5_list)
                    # AR6_mean = mean(AR6_list)
                    # AR7_mean = mean(AR7_list)
                    # MSE_mean = mean(MSE_list)
                    # Distance_mean = mean(Distance_list)
                    # Time_mean = mean(Time_list)
                    minList = min(Distance_pointList, key=len)
                    minDis = len(minList)
                    for i in range(len(Distance_pointList)):
                        delta = abs(len(Distance_pointList[i]) - minDis)
                        del (Distance_pointList[i][0:delta])
                    Distance_point = np.mean(Distance_pointList, axis=0).tolist()
                    # Result_pattern['{}_{}_{}continue'.format(p1,p2,p3)] = ['AR5 = {},AR6 = {},AR7 = {},Distance = {},MSE = {},Time = {}'.format(AR5_mean, AR6_mean, AR7_mean,Distance_mean, MSE_mean,Time_mean)]

                    lineLegendName = args.boxLegendName[ids1]+'_'+args.kpLegendName[ids2]+'_'+args.upDateLegendName[ids3]
                    lineLegendColor = args.LegendColor[ids3]
                    # lineLegendStyle = args.LegendStyle[ids2]
                    # lineLegendMarker = args.LegendMarker[ids3]
                    plt.plot(Distance_point, color=lineLegendColor, linewidth=2, label=lineLegendName)
                    plt.legend(prop={"family": "Times New Roman", "size": 15}, loc='upper left')

                plt.savefig('your path'.format(p1,p2))
                plt.close('all')
        file = open('your path', 'a')
        file.write(str(Result_pattern)+'\n')
        file.close()
        break

if __name__ == "__main__":
    args = parse_args()
    main(args)
