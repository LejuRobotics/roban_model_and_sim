#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
import time
import rospy
import sys
import rospkg
sys.path.append(rospkg.RosPack().get_path('lss_roban'))
from lss_roban.motion_control import *

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import cv2 as cv
import math

import threading
from collections import Iterable

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
x,y,w,h = 320,240,0,0

see_button = False
mask = np.zeros((480,640), dtype='uint8')

bridge = CvBridge()
def D435Colorcallback(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    hsv = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
    lower_red = np.array([0,80,80])
    upper_red = np.array([10,255,255])
    lower_pink = np.array([170,80,80])
    upper_pink = np.array([180,255,255])
    global mask
    mask1 = cv.inRange(hsv, lower_red, upper_red)
    mask2 = cv.inRange(hsv, lower_pink, upper_pink)
    mask = cv.bitwise_or(mask1, mask2)
    opened = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
    closing = cv.morphologyEx(opened, cv.MORPH_CLOSE, np.ones((3,3), np.uint8))
    ret,thresh = cv.threshold(closing,127,255,cv.THRESH_BINARY)
    im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    copy=cv_image.copy()
    global see_button
    if(len(contours)>0):
        contours.sort(key=lambda cnt:cv.contourArea(cnt), reverse=True)
        cnt=contours[0]
        if(cv.contourArea(cnt)<500):
            return
        cv.drawContours(copy, contours, 0, (0,255,0), 1)
        see_button = True
        global x,y,w,h
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(copy,(x,y),(x+w,y+h),(0,255,0),2)
        roi = cv.rectangle(np.zeros((480,640), dtype='uint8'),(x,y),(x+w,y+h),(255),-1)
        mask = cv.bitwise_and(mask,roi)
        #print("area=%d, cx=%f, cy=%f" % (cv.contourArea(cnt), cx, cy))
    else:
        see_button = 0
    cv.imshow("Image window", copy)
    cv.waitKey(25)

def getButtonPos():
    '''
    返回shape=(3)的ndarray
    '''
    if(not see_button):
        return None
    t0 = rospy.get_time()
    cloud = []
    w_x = []
    w_y = []
    #er = cv.erode(mask,np.ones((5,5), dtype='uint8'),iterations = 1)
    if(w<50):
        xs=range(x, x+w)
    else:
        xs=[x+(w-1)*i/50 for i in range(51)]
    if(h<50):
        ys=range(y, y+h)
    else:
        ys=[y+(h-1)*i/50 for i in range(51)]
    for xi in xs:
        for yi in ys:
            if(mask[yi][xi]>0):
                w_x.append(xi)
                w_y.append(yi)
    cloud = getCoordinatesFromDepthImage(w_x, w_y)
    mid = np.median(cloud, axis=1)
    button = cloud[:, (np.abs(cloud[1]-mid[1])<0.1)&(np.abs(cloud[2]-mid[2])<0.1)&(np.abs(cloud[0]-mid[0])<0.02) ]
    button_pos = (np.amin(button,1)+np.amax(button,1))/2
    button_pos[0]=np.median(button[0])
    # 绘制散点图
    '''fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(button[0],button[1],button[2], color='red', s=1)
    ax.scatter(mid[0], mid[1], mid[2], color='green', s=3)
    ax.scatter(button_pos[0], button_pos[1], button_pos[2], color='blue', s=3)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()'''
    t = rospy.get_time()
    print("计算耗时%f s" % (t-t0) )
    return button_pos

class TrackingThread(threading.Thread):
    def __init__(self, pause, over):
        threading.Thread.__init__(self)
        self.pause = pause
        self.over = over
    def run(self):
        rate = rospy.Rate(4)
        rospy.loginfo("视觉追踪线程就绪")
        while(not rospy.is_shutdown()):
            if(self.over.is_set()):
                break
            if(not self.pause.is_set()):
                button_pos = getCoordinateFromDepthImage(x+w/2, 479-(y+h/2))
                if(isinstance(button_pos, Iterable)):
                    visualTrack(button_pos)
            rate.sleep()
        rospy.loginfo("视觉追踪线程结束")

def main():
    rospy.loginfo("开始了")
    rospy.Subscriber("/sim/camera/D435/colorImage", Image, D435Colorcallback)
    setBodyhubNoStand(1)
    setArmMode(0)
    #motionInit()
    walkingInit()
    pause = threading.Event()
    over = threading.Event()
    tracking = TrackingThread(pause, over)
    tracking.start()
    rospy.sleep(1)
    UpdateTargets()
    while(not over.is_set()):
        pause.set()
        button_pos = getButtonPos()
        pause.clear()
        print("button_pos:"+str(button_pos))
        angles1 = ikRightArm(button_pos+np.array([-0.02,0,0]))
        angles2 = ikRightArm(button_pos+np.array([0.01,0,0]))
        if((isinstance(angles1, Iterable) and len(angles1)==3)and(isinstance(angles2, Iterable) and len(angles2)==3)):
            over.set()
            break
        arm_prepare = [0.14, -0.18, button_pos[2]-0.06]

        walk_dist = (button_pos[0]-0.16,button_pos[1]-(-0.14))
        print("walk_dist:"+str(walk_dist))
        w_thread = WalkingThread(walk_dist[0], walk_dist[1], 0)
        w_thread.start()
        bodyMoveTo(right_arm=[arm_prepare], count=50)
        w_thread.join()
    tracking.join()
    #raw_input("触碰...")
    button_pos = getButtonPos()
    linearTrajMoveTo(right_arm = [button_pos+np.array([-0.02,0,0])], divide = 10)
    linearTrajMoveTo(right_arm = [button_pos+np.array([+0.01,0,0])], divide = 10)
    linearTrajMoveTo(right_arm = [button_pos+np.array([-0.02,0,0])], divide = 10)
    #raw_input("完成了...")
    #print("finish")



if __name__ == '__main__':
    
    def rosShutdownHook():
        setBodyhubNoStand(0)
        setArmMode(1)
        resetBodyhub()
        rospy.signal_shutdown('node_close')

    rospy.init_node('task02_press_button', anonymous=True)
    rospy.on_shutdown(rosShutdownHook)    
    main()