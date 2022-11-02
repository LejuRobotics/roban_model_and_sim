#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
from collections import Iterable
import threading
from math import *
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import matplotlib.pyplot as plt
from tf.transformations import *
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from lss_roban.motion_control import *
import rospy
import sys
import rospkg
sys.path.append(rospkg.RosPack().get_path('lss_roban'))

x, y, w, h = 320, 240, 0, 0
mask_select = np.zeros((480, 640), dtype='uint8')
color_type = 4
see_trash = False
# 4 表示 未选择, 0 表示 蓝色, 1 表示 红色, 2 表示黄色, 3 表示 黑色

color_mask = [
    lambda img: cv.inRange(cv.cvtColor(img,cv.COLOR_BGR2HSV), np.array([96, 220, 140]), np.array([104,255,255])),
    lambda img: cv.bitwise_or(cv.inRange(cv.cvtColor(img,cv.COLOR_BGR2HSV), np.array([0, 160, 80]), np.array([10,255,255])),
        cv.inRange(cv.cvtColor(img,cv.COLOR_BGR2HSV), np.array([170, 160, 80]), np.array([180,255,255]))),
    lambda img: cv.inRange(cv.cvtColor(img,cv.COLOR_BGR2HSV), np.array([20,200,80]), np.array([40,255,255])),
    lambda img: cv.bitwise_or(cv.inRange(img, np.array([55, 55, 55]), np.array([75,75,75])),
        cv.inRange(img, np.array([100, 100, 100]), np.array([120,120,120]))),
]

see_trash=False 
bridge = CvBridge()
global is_want
is_want = 'trash'

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
                mat_pos = getCoordinateFromDepthImage(x+w/2, 479-(y+h/2))
                if(isinstance(mat_pos, Iterable) and see_trash):
                    visualTrack(mat_pos)
            rate.sleep()
        rospy.loginfo("视觉追踪线程结束")

class UpdateThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True

    def run(self):
        rate = rospy.Rate(15)
        matQ = []
        rospy.loginfo("自身稳态检测线程就绪")
        while(not rospy.is_shutdown()):
            mat_pos = getCoordinateFromDepthImage(x+w/2, 479-(y+h/2))
            if(isinstance(mat_pos, Iterable)):
                matQ.append(mat_pos)
                if(len(matQ)>20):
                    matQ.pop(0)
                global stable
                if(np.amax(np.ptp(matQ, axis=0))/(mat_pos[0]+0.1)>0.015):
                    stable=False
                else:
                    stable=True
            rate.sleep()
        rospy.loginfo("稳态检测线程结束")

def cnt_pos(cnt):
    x, y, w, h = cv.boundingRect(cnt)
    center_pos = getCoordinateFromDepthImage(x+w/2, 479-(y+h/2))
    return center_pos

def getTargetObjPos(stupid = False):
    '''
    返回shape=(3)的ndarray
    当stupid, 不判断是不是想要的垃圾
    '''
    if(not (see_trash or stupid)):
        return None
    t0 = rospy.get_time()
    cloud = []
    w_x = []
    w_y = []
    if(w < 50):
        xs = range(x, x+w)
    else:
        xs = [x+(w-1)*i/50 for i in range(51)]
    if(h < 50):
        ys = range(y, y+h)
    else:
        ys = [y+(h-1)*i/50 for i in range(51)]
    for xi in xs:
        for yi in ys:
            if(mask_select[yi][xi] > 0):
                w_x.append(xi)
                w_y.append(yi)
    cloud = getCoordinatesFromDepthImage(w_x, w_y)
    BETWEEN = np.median(cloud, axis=1)
    if(is_want == 'trash'):
        mat = cloud[:, (np.abs(cloud[1]-BETWEEN[1]) < 0.03) & (np.abs(cloud[2]-BETWEEN[2]) < 0.07)
                    & (np.abs(cloud[0]-BETWEEN[0]) < 0.03)]
    elif(is_want == 'bin'):
        mat = cloud[:, (np.abs(cloud[1]-BETWEEN[1]) < 0.2) & (np.abs(cloud[2]-BETWEEN[2]) < 0.5)
                    & (np.abs(cloud[0]-BETWEEN[0]) < 0.2)]
    if(len(mat[0])==0):
        return None
    mat_pos = (np.amax(mat, axis=1)+np.amin(mat, axis=1))*0.5
    # 绘制散点图
    '''
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(mat[0], mat[1], mat[2], color='blue', s=1)
    #ax.scatter(BETWEEN[0], BETWEEN[1], BETWEEN[2], color='green', s=3)
    ax.scatter(mat_pos[0], mat_pos[1], mat_pos[2], color='red', s=5)
    #ax.scatter(cloud[0],cloud[1],cloud[2],color='blue',s=3)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()
    '''
    t = rospy.get_time()
    #print("计算耗时%f s" % (t-t0))
    print(mat_pos)
    return mat_pos

color_callback_lock=threading.Lock()
def D435Colorcallback(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    color_callback_lock.acquire()
    global see_trash,color_type,see_trash
    global mask_select
    for i in (range(4) if color_type==4 else [color_type]):
        mask_select= color_mask[i](cv_image)
        opened = cv.morphologyEx(mask_select, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
        closing = cv.morphologyEx(opened, cv.MORPH_CLOSE,
                                np.ones((3, 3), np.uint8))
        ret, thresh = cv.threshold(closing, 127, 255, cv.THRESH_BINARY)
        im2, contours, hierarchy = cv.findContours(
            thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        copy = cv_image.copy()
        copy = cv.cvtColor(copy, cv.COLOR_BGR2HSV)
        if(is_want == 'trash'):
            condition = lambda cnt:(cnt_pos(cnt)[2]>-0.18 and cnt_pos(cnt)[0]<1)
        elif(is_want == 'bin'):
            condition = lambda cnt:cnt_pos(cnt)[2]<-0.25
        contours=[cnt for cnt in contours if condition(cnt)]
        if(len(contours) > 0):
            contours.sort(key=lambda cnt: cv.contourArea(cnt), reverse=True)
            cnt = contours[0]
            x0, y0, w0, h0 = cv.boundingRect(cnt)
            if(w0 < 20 or cv.contourArea(cnt) < 300):
                continue
            cv.drawContours(copy, contours, 0, (0, 255, 0), 1)
            color_type=i
            global x, y, w, h
            x, y, w, h = cv.boundingRect(cnt)
            see_trash = True
            cv.rectangle(copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = cv.rectangle(np.zeros((480, 640), dtype='uint8'),
                            (x, y), (x+w, y+h), (255), -1)
            cv.imshow("Image window",copy)
            cv.waitKey(25)
            break
        else:
            see_trash=False
    cv.imshow("Image window", copy)
    cv.waitKey(25)
    color_callback_lock.release()

def main():
    rospy.loginfo("开始了")
    sub=rospy.Subscriber("/sim/camera/D435/colorImage", Image, D435Colorcallback)
    setBodyhubNoStand(1)
    setArmMode(0)
    walkingInit()
    gripperCtrl(right=90)
    rospy.sleep(0.5)
    UpdateTargets()
    headCtrl(yaw=-20, pitch=55)
    global color_type
    if(color_type == 4):
        headCtrl(yaw=40, pitch=25)
    pause = threading.Event()
    goto_trash = threading.Event()
    tracking = TrackingThread(pause, goto_trash)
    tracking.start()
    updating = UpdateThread()
    updating.start()
    waitForActionExecute()
    print('color type:%d' % color_type)
    #------------------------------------------------------------------------------------------------
    #去抓垃圾
    #trash_pos = [ 0.20374397 -0.15539208 -0.14117675]
    rh = [-0.033929,-0.05522365,-0.01340744]
    while(not goto_trash.is_set()):
        pause.set()
        yaw0 = measures[20]
        pitch0 = measures[21]
        if(not see_trash):
            headCtrl(yaw=yaw0, pitch=pitch0+30)
        if(not see_trash):
            headCtrl(yaw=yaw0+30, pitch=pitch0+30)
        if(not see_trash):
            headCtrl(yaw=yaw0+30, pitch=pitch0-30)
        if(not see_trash):
            headCtrl(yaw=yaw0-30, pitch=pitch0-30)
        if(not see_trash):
            headCtrl(yaw=yaw0-30, pitch=pitch0+30)
        pause.clear()
        trash_pos = getTargetObjPos()
        while(not isinstance(trash_pos, Iterable)):
            rospy.sleep(0.1)
            trash_pos = getTargetObjPos()
        print("trash_pos:"+str(trash_pos))
        angles = ikRightArm(trash_pos+[0,0,0.14+0.06], rh)
        if(isinstance(angles, Iterable) and len(angles) == 3):
            #等稳定下来再测一次
            while(not stable):
                print('等不晃了')
                rospy.sleep(0.2)
            angles = ikRightArm(trash_pos+[0,0,0.14], rh)
            if(isinstance(angles, Iterable) and len(angles) == 3):
                goto_trash.set()
                break
            else:
                print('看来还是有点问题，再挪一挪')
        arm_prepare = [0.14, -0.18, trash_pos[2]+0.14]
        walk_dist = (trash_pos[0]-0.16, trash_pos[1]-(-0.14))
        print("walk_dist:"+str(walk_dist))
        w_thread = WalkingThread(walk_dist[0], walk_dist[1], 0)
        w_thread.start()
        gripperCtrl(right=90)
        bodyMoveTo(right_arm=[arm_prepare, rh], count=30)
        w_thread.join()
    #goto_trash.set()
    motionInit()
    linearTrajMoveTo(torso=[(0,0,-0.14)])
    trash_pos = getTargetObjPos()
    tracking.join()
    print('头', targets[20:22])
    global is_want
    is_want = 'bin'
    rospy.logerr('抓取动作:%s', str(trash_pos+(0,-0.01,0)))
    linearTrajMoveTo(right_arm=[trash_pos+(0,-0.01, 0), rh], divide=10)
    gripperCtrl(right=20)
    linearTrajMove(right_arm=[(0,+0.01, 0), rh], divide=5)
    waitForActionExecute()
    #------------------------------------------------------------------------------------------------
    #去抓垃圾
    rospy.loginfo('开始固定走路段')
    t0=sim_time.time()
    name=['蓝','红','黄','灰']
    rospy.logerr('t0=%f s, 颜色%s',t0,name[color_type])
    linearTrajMoveTo(torso=[[0,0,0.14]], right_arm=[(0.14, -0.16, -0.06)])
    headCtrl(pitch=25)
    walkingInit()
    sendGaitCommand(0,0,-30)
    sendGaitCommand(0,0,-30)
    sendGaitCommand(0,0,-30)
    #sendGaitCommand(0,0.06,0)
    #headCtrl(yaw=-45)
    rospy.sleep(0.5)
    #-----------------------------------------------------------------------------------------------------------------------------
    #去扔垃圾
    pause = threading.Event()
    goto_bin = threading.Event()
    tracking = TrackingThread(pause, goto_bin)
    tracking.start()
    #sendGaitCommand()

    #sendGaitCommand(0.05,0,0)
    sendGaitCommand(0.02,0,0)
    sendGaitCommand(0.08,0,0)
    sendGaitCommand(0.10,0,0)
    sendGaitCommand(0.10,0,0)
    #sendGaitCommand(0,0,10)
    # walk_dist = (0.5 0)
    #walkTheDistance(walk_dist[0], walk_dist[1], 0)
    waitForWalkingDone()
    if(color_type==1):
        sendGaitCommand(0.10,0,0)
    if(color_type==2):
        sendGaitCommand(0.10,0,0)
        sendGaitCommand(0.10,0,0)
    if(color_type==3):
        sendGaitCommand(0.10,0,0)
        sendGaitCommand(0.10,0,0)
        sendGaitCommand(0.10,0,0)
        sendGaitCommand(0.05,0,0)
    for i in range(4):
        sendGaitCommand(0.1,0,9.8)
        sendGaitCommand(0.1,0,9.8)
        print("hh")
    print("hello")
    #sendGaitCommand(0.04,0,2)   
    if(color_type==3):
        sendGaitCommand(0,0,8)
        print("z")  
    for i in range(4):
        sendGaitCommand(0.040,0,0)
    print("hhhhhh")
    
    while(not isinstance(trash_pos, Iterable)):
        print('不对劲')
        rospy.sleep(0.1)
        trash_pos = getTargetObjPos()
    rospy.loginfo('固定阶段已经结束了')
    t1=sim_time.time()
    rospy.logerr('t1=%f s, 阶段性花费%f s',t1, t1-t0)
    #---------------------------------------------------------------------------------
    waitForWalkingDone()
    above = 0.15
    while(not goto_bin.is_set()):
        pause.set()
        yaw0 = measures[20]
        pitch0 = measures[21]
        if(not see_trash):
            headCtrl(yaw=yaw0, pitch=pitch0+30)
        if(not see_trash):
            headCtrl(yaw=yaw0+30, pitch=pitch0+30)
        if(not see_trash):
            headCtrl(yaw=yaw0+30, pitch=pitch0-30)
        if(not see_trash):
            headCtrl(yaw=yaw0-30, pitch=pitch0-30)
        if(not see_trash):
            headCtrl(yaw=yaw0-30, pitch=pitch0+30)
        pause.clear()
        bin_pos = getTargetObjPos()
        while(not isinstance(bin_pos, Iterable)):
            rospy.sleep(0.1)
            bin_pos = getTargetObjPos()
        print("bin_pos:"+str(bin_pos))
        angles = ikRightArm(bin_pos+[-0.04,0,0.14+above], rh)
        if(isinstance(angles, Iterable) and len(angles) == 3):
            #等稳定下来再测一次
            while(not stable):
                print('等不晃了')
                rospy.sleep(0.2)
            angles = ikRightArm(bin_pos+[-0.04,0,0.14+above], rh)
            if(isinstance(angles, Iterable) and len(angles) == 3):
                goto_bin.set()
                break
            else:
                print('看来还是有点问题，再挪一挪')
        walk_dist = (bin_pos[0]-0.20, bin_pos[1]-(-0.14))
        print("walk_dist:"+str(walk_dist))
        w_thread = WalkingThread(walk_dist[0], walk_dist[1], 0)
        w_thread.start()
        w_thread.join()
    t2=sim_time.time()
    rospy.logerr('t2=%f s, 共花费%f s',t0, t2-t0)
    goto_bin.set()
    motionInit()
    bin_pos = getTargetObjPos()
    linearTrajMoveTo(right_arm=[bin_pos+(-0.04,0,0.14+above)], torso=[(0.04,0,-0.14)])
    gripperCtrl(right=25)
    linearTrajMove(right_arm=[(0, 0.003, 0)], divide=5)
    tracking.join()
    



if __name__ == '__main__':

    def rosShutdownHook():
        setBodyhubNoStand(0)
        setArmMode(1)
        resetBodyhub()
        rospy.signal_shutdown('node_close')
 
    rospy.init_node('test_motion_control', anonymous=True)
    rospy.on_shutdown(rosShutdownHook)
    main()

