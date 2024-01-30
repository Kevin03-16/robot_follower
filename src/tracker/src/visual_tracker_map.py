#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 
import sys
import torch
import pyrealsense2 as rs
import threading
from actionlib import GoalStatus
import actionlib
# from ros_tracker.action import last_move as LastMoveAction
import random
import time
import math
import numpy as np
# message_filters是一个用于roscpp和rospy的实用程序库。 它集合了许多的常用的消息“过滤”算法。
# 消息过滤器message_filters类似一个消息缓存，当消息到达消息过滤器的时候，可能并不会立即输出，而是在稍后的时间点里满足一定条件下输出。
import message_filters
from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device
# from yolov5.utils.downloads import attempt_download
from yolov5.utils.general import LOGGER, non_max_suppression, scale_boxes, xyxy2xywh
from yolov5.utils.torch_utils import select_device
from yolov5.utils.dataloaders import letterbox

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import rospy
from std_msgs.msg import String as StringMsg
from geometry_msgs.msg import PointStamped, PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ros_tracker.msg import position as PositionMsg

import tf

device = select_device('0') if torch.cuda.is_available() else "cpu"

def init_sort(config_deepsort, deep_sort_weights, current_path):
# initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    # attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(model_path=current_path + cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    return deepsort

def init_yolo(yolo_weights, half):
    print(device)
    # select_device()函数是torch_utils中的函数，将程序装载至对应的位置
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load(yolo_weights, device=device)  
    if half:
        return model.half().eval()
    else:
        return model.eval()
    
class visualTracker():

    def __init__(self, deepsort, yolo, half, augment, conf_thres, iou_thres, classes, agnostic_nms, max_det):
        rospy.loginfo("Start PeopleTracker Init process...")
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.half = half
        self.deepsort = deepsort
        self.yolo = yolo
        # self.realsense = realsense
        self.imgsz = 640
        self.names = self.yolo.module.names if hasattr(self.yolo, 'module') else self.yolo.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]  

        self.rate = rospy.Rate(5)
        self.bridge = CvBridge()
        self.pictureHeight = rospy.get_param('~pictureDimensions/pictureHeight')
        self.pictureWidth = rospy.get_param('~pictureDimensions/pictureWidth')
        vertAngle =rospy.get_param('~pictureDimensions/verticalAngle')
        horizontalAngle =  rospy.get_param('~pictureDimensions/horizontalAngle')
        # precompute tangens since thats all we need anyways:
        self.tanVertical = np.tan(vertAngle)
        self.tanHorizontal = np.tan(horizontalAngle)
        self.detla_x, self.delta_y = None, None
        self.lastPosition =None
        self.cam_image = None
        self.arrivedLastPos = False
        self.tf_listener = tf.TransformListener()
        # one callback that deals with depth and rgb at the same time
        image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image) #  frame_id: "camera_rgb_optical_frame"
        dep_sub = message_filters.Subscriber("/camera/depth/image_raw", Image) #  frame_id: "camera_rgb_optical_frame"
        '''
        如果需要绝对时间同步，那么需要时间戳相同
        TimeSynchronizer过滤器通过包含在其头中的时间戳来同步输入通道
        并以单个回调的形式输出它们需要相同数量的通道
        '''
        self.timeSynchronizer = message_filters.ApproximateTimeSynchronizer([image_sub, dep_sub], 10, 0.5)
        self.timeSynchronizer.registerCallback(self.detector_tracker)
        self.positionPublisher = rospy.Publisher('/object_tracker/current_position', PositionMsg, queue_size=3)
        self.infoPublisher = rospy.Publisher('/object_tracker/info', StringMsg, queue_size=3)
        self.image_pub = rospy.Publisher("cv_bridge_image", Image, queue_size=1)

    def publishPosition(self, pos):
        # calculate the angles from the raw position
        angleX = self.calculateAngleX(pos)
        angleY = self.calculateAngleY(pos)

        # publish the position (angleX, angleY, distance)
        posMsg = PositionMsg(angleX, angleY, pos[0][0], pos[0][1], pos[1])
        self.positionPublisher.publish(posMsg)

    def checkPosPlausible(self, pos):
        '''Checks if a position is plausible. i.e. close enough to the last one.'''

        # for the first scan we cant tell
        # if self.lastPosition is None:
        #     return False
        ((centerX, centerY), dist)=pos	
        if np.isnan(dist):
            return False
        if self.lastPosition is not None:
            # unpack positions
            ((PcenterX, PcenterY), Pdist)=self.lastPosition
            self.detla_x = dist - Pdist
            self.delta_y= dist * math.tan(self.calculateAngleX(pos)) - Pdist * math.tan(self.calculateAngleX(self.lastPosition))
            # distance changed to much
            # if abs(dist-Pdist)>0.5:
            #     print("2")
            #     return False
            # # location changed to much (5 is arbitrary)
            # if abs(x_dir)>(self.pictureWidth /5):
            #     print("3")
            #     return False
            # if abs(y_dir)>(self.pictureHeight/5):
            #     print("4")
            #     return False
        return True

    def calculateAngleX(self, pos):
        '''calculates the X angle of displacement from straight ahead'''
        centerX = pos[0][0]
        displacement = 2*centerX/self.pictureWidth-1
        # displacement = (2*centerX/640 -1) * (self.pictureWidth / 640)
        angle = -1*np.arctan(displacement*self.tanHorizontal)
        return angle

    # 根据目标在图像中的Y坐标位置，计算目标相对于图像中心的垂直方向的角度
    def calculateAngleY(self, pos):
        centerY = pos[0][1]
        # 计算目标在图像中的Y坐标相对于图像高度的归一化位移。将Y坐标映射到范围[-1, 1]。
        displacement = 2*centerY/self.pictureHeight-1
        # displacement = (2*centerY/640 -1) * (self.pictureHeight / 640)
        # 使用反正切函数（arctan）计算目标相对于图像中心的垂直方向角度
        # tan(视角/2) = 图像尺寸的一半 / 焦距
        angle = -1*np.arctan(displacement*self.tanVertical)
        return angle

    def analyseContour(self, box, depthFrame):
        '''Calculates the centers coordinates and distance for a given contour

        Args:
            contour (opencv contour): contour of the object
            depthFrame (numpy array): the depth image
        
        Returns:
            centerX, centerY (doubles): center coordinates
            averageDistance : distance of the object
        '''
        # # get a rectangle that completely contains the object
        # centerRaw, size, rotation = cv2.minAreaRect(contour)

        # # get the center of that rounded to ints (so we can index the image)
        # center = np.round(centerRaw).astype(int)

        center = [int(box[0] + box[2])//2, int(box[1] + box[3])//2] #确定索引深度的中心像素位置
        cv2.circle(self.cam_image, (center[0], center[1]), 4, (255, 255, 255), 5)#标出中心点
        size = [int(box[2] - box[0]), int(box[3] - box[1])]
        # find out how far we can go in x/y direction without leaving the object (min of the extension of the bounding rectangle/2 (here 3 for safety)) 
        minSize = int(min(size)/3)

        # get all the depth points within this area (that is within the object)
        depthObject = depthFrame[(center[1]-minSize):(center[1]+minSize), (center[0]-minSize):(center[0]+minSize)]

        # get the average of all valid points (average to have a more reliable distance measure)
        depthArray = depthObject[~np.isnan(depthObject)]
        averageDistance = np.mean(depthArray)

        if len(depthArray) == 0:
            rospy.logwarn('empty depth array. all depth values are nan')

        return (center, averageDistance)
    
    def get_mid_pos(self, box):
        mid_pixel = [(box[0] + box[2])//2, (box[1] + box[3])//2] #确定索引深度的中心像素位置
        # 注意mid_pixel是float，但是get_distance()的参数需要是int
        [ux, uy] = [int(mid_pixel[0]), int(mid_pixel[1])]
        camera_xyz = self.realsense.get_xyz(ux, uy)
        cv2.circle(self.cam_image, (ux,uy), 4, (255, 255, 255), 5)#标出中心点
        # cv2.putText(canvas, str(camera_xyz), (ux+20, uy+10), 0, 1,
        #                             [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)#标出坐标
        return ([camera_xyz[0], camera_xyz[1]], camera_xyz[2])

    def image_process(self):
        im = self.cam_image[:, :, 0:3]
        '''
        在常用的目标检测算法中，不同的图片长宽都不相同，因此常用的方式是将原始图片统一缩放到一个标准尺寸，再送入检测网络中
        '''
        img = letterbox(im, self.imgsz)[0]  #channel=3
        # self.pictureHeight, self.pictureWidth = im.shape[:2]
        '''
            RGB 通常用于图像编辑和显示应用程序，顺序为红色、绿色和蓝色。
            BGR 通常用于图像处理应用程序，顺序为蓝色、绿色和红色。
            使用 Python 处理图像文件时，OpenCV 库（cv2）在读取图像时默认使用 BGR 颜色空间，而 PIL 库使用 RGB 颜色空间    
        '''
        img = img[:, :, ::-1].transpose(2, 0, 1)  #BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)  #channel = imgsz

        img = torch.from_numpy(img).to(device)
        #uint8 to fp16/32
        img = img.half() if self.half else img.float()  
        #0 - 255 to 0.0 - 1.0
        img /= 255.0   #512
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img, im
    
    def detector_tracker(self, image_data, depth_data):
        '''
        32位深度图即一个像素值占四个byte，是32位浮点数，单位是米
        16位深度图即一个像素值占两个byte，是16位整数，单位是毫米
        '''
        try:
            depthFrame = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding='32FC1')#"32FC1"32位深度图
            self.cam_image = self.bridge.imgmsg_to_cv2(image_data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        img, im = self.image_process()
        pred = self.yolo(img, augment=self.augment)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        if len(pred[0]) == 0:
            if self.lastPosition is None:
                self.infoPublisher.publish('No_people')
            else:
                self.infoPublisher.publish('Loss_target')
        else:
            self.infoPublisher.publish('People_detected')
            self.tracker(pred, im, img, depthFrame)
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.cam_image, "bgr8"))
            except CvBridgeError as e:
                print(e)

    def tracker(self, pred, im, img, depthFrame):
        # i表示 batch det表示五个预测框
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                s, im0 = '', im.copy()
                s += '%gx%g ' % img.shape[2:]  
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                # Convert from [x1, y1, x2, y2] to [x, y, w, h] 
                # where xy1=top-left, xy2=bottom-right, x=x_center, y=y_center
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                # pass detections to deepsort
                outputs = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)): 
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        c = int(cls)  # integer class
                        label = f'{id} {self.names[c]} {conf:.2f}'
                        cv2.rectangle(self.cam_image, (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3])), (0, 255, 0), 2)
                        cv2.putText(self.cam_image, str(label), (int(bboxes[0]), int(bboxes[1])), 0, 5e-3 * 200, (0, 255, 0), 2)  
                        pos = self.analyseContour(bboxes,depthFrame)
                        # if it's the first one we found it will be the fall back for the next scan if we don't find a plausible one
                        # check if the position is plausible
                        if self.checkPosPlausible(pos):
                            self.lastPosition = pos
                            self.publishPosition(pos)
                            return
        # cv2.imshow("YOLO + SORT", self.cam_image)
        # cv2.waitKey(1)

if __name__ == '__main__':
    current_path = sys.path[0]
    config_deepsort = current_path + "/deep_sort_pytorch/configs/deep_sort.yaml"
    deep_sort_weights = current_path + "/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
    yolo_weights = current_path + "/yolov5/weights/yolov5s.pt"

    half = True
    deepsort = init_sort(config_deepsort, deep_sort_weights, current_path)
    yolo = init_yolo(yolo_weights, half)
    augment = False
    conf_thres = 0.4
    iou_thres = 0.5
    classes = 0
    agnostic_nms = False
    max_det = 1000
    rospy.init_node('visual_tracker', log_level=rospy.INFO)
    # realsense = Realsense()
    tracker=visualTracker(deepsort, yolo, half, augment, conf_thres, iou_thres, classes, agnostic_nms, max_det)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down cv_bridge_test node")
        cv2.destroyAllWindows()
    # tracking = TrackerPeople()
    # tracking.loop()
    # cv2.destroyAllWindows()



