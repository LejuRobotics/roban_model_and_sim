#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
from collections import Iterable
import threading
import math
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import matplotlib.pyplot as plt
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from lss_roban.motion_control import *
import time
import rospy
import sys
import rospkg
sys.path.append(rospkg.RosPack().get_path('lss_roban'))


x, y, w, h = 320, 240, 0, 0

see_handle = False
# cup_bias = [0, 0.01, -0.06]
mask = np.zeros((480, 640), dtype='uint8')

bridge = CvBridge()


def D435Colorcallback(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    hsv = cv.cvtColor(cv_image, cv.COLOR_BGR2HSV)
    lower_blue = np.array([90, 80, 80])
    upper_blue = np.array([110, 255, 255])
    global mask
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    opened = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    closing = cv.morphologyEx(opened, cv.MORPH_CLOSE,
                              np.ones((3, 3), np.uint8))
    ret, thresh = cv.threshold(closing, 127, 255, cv.THRESH_BINARY)
    im2, contours, hierarchy = cv.findContours(
        thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    copy = cv_image.copy()
    global see_handle
    if(len(contours) > 0):
        contours.sort(key=lambda cnt: cv.contourArea(cnt), reverse=True)
        cnt = contours[0]
        if(cv.contourArea(cnt) < 500):
            return
        cv.drawContours(copy, contours, 0, (0, 0, 255), 1)
        see_handle = True
        global x, y, w, h
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi = cv.rectangle(np.zeros((480, 640), dtype='uint8'),
                           (x, y), (x+w, y+h), (255), -1)
        # mask = cv.bitwise_and(mask,roi)
        # print("area=%d, cx=%f, cy=%f" % (cv.contourArea(cnt), cx, cy))
    else:
        see_handle = 0
    cv.imshow("Image window", copy)
    cv.waitKey(25)


def getHandlePos():
    '''
    返回shape=(3)的ndarray
    '''
    if(not see_handle):
        return None
    t0 = rospy.get_time()
    cloud = []
    w_x = []
    w_y = []
    # er = cv.erode(mask,np.ones((5,5), dtype='uint8'),iterations = 1)
    if(w < 50):
        xs = range(x, x+w)
    else:
        xs = [x+(w-1)*i/50 for i in range(51)]
    if(h < 50):
        ys = range(y, y+h)
    else:
        ys = [y+(h-1)*i/50 for i in range(51)]
    # print(xs)
    # print(ys)
    for xi in xs:
        for yi in ys:
            if(mask[yi][xi] > 0):
                w_x.append(xi)
                w_y.append(yi)
    cloud = getCoordinatesFromDepthImage(w_x, w_y)
    BETWEEN = np.median(cloud, axis=1)
    handle = cloud[:, (np.abs(cloud[1]-BETWEEN[1]) < 0.1) & (cloud[2] >= 0.03)
                   & (np.abs(cloud[0]-BETWEEN[0]) < 0.1)]
    handle_pos = np.median(handle, axis=1)
    # 绘制散点图
    '''
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(handle[0], handle[1], handle[2], color='red', s=1)
    ax.scatter(BETWEEN[0], BETWEEN[1], BETWEEN[2], color='green', s=3)
    ax.scatter(handle_pos[0], handle_pos[1], handle_pos[2], color='blue', s=3)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()
    '''
    t = rospy.get_time()
    print("计算耗时%f s" % (t-t0))
    return handle_pos


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
                handle_pos = getCoordinateFromDepthImage(x+w/2, 479-(y+h/2))
                if(isinstance(handle_pos, Iterable)):
                    visualTrack(handle_pos)
            rate.sleep()
        rospy.loginfo("视觉追踪线程结束")


def main():
    rospy.loginfo("开始了")
    rospy.Subscriber("/sim/camera/D435/colorImage", Image, D435Colorcallback)
    setBodyhubNoStand(1)
    setArmMode(0)
    # motionInit()
    walkingInit()
    # 首先找到垫子的位置
    if(not see_handle):
        headCtrl(yaw=-30, pitch=15)
        waitForActionExecute()
    if(not see_handle):
        headCtrl(yaw=30, pitch=15)
        waitForActionExecute()
    pause = threading.Event()
    over = threading.Event()
    tracking = TrackingThread(pause, over)
    tracking.start()
    rospy.sleep(1)  # 暂停ros
    UpdateTargets()
    while(not over.is_set()):
        pause.set()
        yaw0 = measures[20]
        pitch0 = measures[21]
        if(not see_handle):
            headCtrl(yaw=yaw0, pitch=pitch0+30)
            waitForActionExecute()
        if(not see_handle):
            headCtrl(yaw=yaw0+30, pitch=pitch0+30)
            waitForActionExecute()
        if(not see_handle):
            headCtrl(yaw=yaw0+30, pitch=pitch0-30)
            waitForActionExecute()
        if(not see_handle):
            headCtrl(yaw=yaw0-30, pitch=pitch0-30)
            waitForActionExecute()
        if(not see_handle):
            headCtrl(yaw=yaw0-30, pitch=pitch0+30)
            waitForActionExecute()
        pause.clear()
        pause.set()
        handle_pos = getHandlePos()
        pause.clear()
        print("handle_pos:"+str(handle_pos))
        angles1 = ikRightArm(handle_pos + np.array([0, 0, 0]), "BT")
        angles2 = ikRightArm(handle_pos + np.array([0, 0.08, 0]), "BT")
        if((isinstance(angles1, Iterable) and len(angles1) == 3) and (isinstance(angles2, Iterable) and len(angles2) == 3)):
            over.set()
            break

        arm_prepare = [0.15, -0.2, handle_pos[2]+0.06]
        gripperCtrl(right=45)
        walk_dist = (handle_pos[0]-0.16, handle_pos[1]-(-0.18))
        print("walk_dist:"+str(walk_dist))
        w_thread = WalkingThread(walk_dist[0], walk_dist[1], 0)
        w_thread.start()
        bodyMoveTo(right_arm=[arm_prepare, "BT"], count=10)
        w_thread.join()
    tracking.join()

    rospy.loginfo("伸手")
    linearTrajMoveTo(
        right_arm=[handle_pos + np.array([0, 0, 0]), "BT"], divide=10)
    # raw_input("夹...")
    rospy.loginfo("夹住")
    gripperCtrl(right=10)
    rospy.loginfo("关闭")
    # raw_input("关...")
    linearTrajMove(
        right_arm=[(0, 0.1, 0), "BT"], divide=10)

    rospy.loginfo("抽手")
    linearTrajMove(right_arm=[(0, 0, 0.06), "BT"], divide=10)
    linearTrajMove(right_arm=[(-0.03, -0.04, 0), "BT"], divide=10)
    gripperCtrl(right=-25)
    stupid = np.zeros((22,))
    stupid[12:15] = (0, -70, -15)
    ra = getTfMat('RH_FE', joints=stupid)[:3, 3]
    linearTrajMoveTo(right_arm=[ra], divide=10)

    # raw_input("完成了...")


if __name__ == '__main__':

    def rosShutdownHook():
        setBodyhubNoStand(0)
        setArmMode(1)
        resetBodyhub()
        rospy.signal_shutdown('node_close')

    rospy.init_node('test_motion_control', anonymous=True)
    rospy.on_shutdown(rosShutdownHook)
    main()
