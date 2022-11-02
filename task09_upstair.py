#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
import rospy
import sys
import rospkg
sys.path.append(rospkg.RosPack().get_path('lss_roban'))
from lss_roban.motion_control import *

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import cv2 as cv
from lss_roban import sim_time
import threading
from collections import Iterable

x,y,w,h = 320,240,0,0

see_stair = False
mask = np.zeros((480,640), dtype='uint8')
waggle=999
edgeQ=[]
bridge = CvBridge()
def D435Colorcallback(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    lower_gray = np.array([0, 0, 15])
    upper_gray = np.array([255, 70, 70])

    global mask
    blur = cv.blur(cv_image, (5, 5))
    mask = cv.inRange(cv.cvtColor(blur, cv.COLOR_BGR2HSV), lower_gray, upper_gray)
    opened = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
    closing = cv.morphologyEx(opened, cv.MORPH_CLOSE,
                              np.ones((3, 3), np.uint8))
    ret, mask = cv.threshold(closing, 127, 255, cv.THRESH_BINARY)
    im2, contours, hierarchy = cv.findContours(
        mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    copy = cv_image.copy()
    copy  = cv.cvtColor(copy, cv.COLOR_BGR2HSV)
    global see_stair
    if(len(contours) > 0):
        contours.sort(key=lambda cnt: cv.contourArea(cnt), reverse=True)
        cnt = contours[0]
        if(cv.contourArea(cnt) < 25):
            return
        cv.drawContours(copy, contours, 0, (0, 255, 0), 2)
        see_stair = True
        global x, y, w, h
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(copy, (x, y), (x+w, y+h), (0, 255, 0), 4)
        # print("area=%d, cx=%f, cy=%f" % (cv.contourArea(cnt), cx, cy))
    else:
        see_stair = 0
    cv.imshow("Image window", copy)
    cv.waitKey(25)

def getBottomPos():
    '''
    返回shape=(3)的ndarray
    '''
    if(not see_stair):
        return None
    t0 = rospy.get_time()
    cloud = []
    w_x = []
    w_y = []
    if(w < 80):
        xs = range(x, x+w)
    else:
        xs = [x+(w-1)*i/50 for i in range(51)]
    if(h < 80):
        ys = range(y, y+h)
    else:
        ys = [y+(h-1)*i/50 for i in range(51)]
    for xi in xs:
        for yi in ys:
            if(mask[yi][xi] > 0):
                w_x.append(xi)
                w_y.append(yi)
    cloud = getCoordinatesFromDepthImage(w_x, w_y)
    top = cloud[:, np.abs(cloud[2]-(-0.31))<0.02]
    bottom = cloud[:, np.abs(cloud[2]-(-0.45))<0.03]
    if(len(cloud[0])==0 or len(top[0])==0 or len(bottom[0])==0):
        return None
    top_p = np.amin(top[0]), np.amax(top[1])-0.25, np.median(top[2])
    bottom_p = np.amin(bottom[0]), np.amax(cloud[1])-0.22, np.median(bottom[2])
    # 绘制散点图
    '''
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(cloud[0], cloud[1], cloud[2], color='blue', s=1)
    ax.scatter(bottom[0], bottom[1], bottom[2], color='red', s=3)
    ax.scatter(top[0], top[1], top[2], color='red', s=3)
    ax.scatter(bottom_p[0], bottom_p[1], bottom_p[2], color='green', s=5)
    ax.scatter(top_p[0], top_p[1], top_p[2], color='green', s=5)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()
    '''
    t = rospy.get_time()
    print("计算耗时%f s" % (t-t0))
    if(bottom_p[0] > top_p[0] - 0.4):
        return None
    return bottom_p

def getTopEdge():
    '''
    返回(degrees(theta), dist, height)
    '''
    if(not see_stair):
        return None
    t0 = rospy.get_time()
    cloud = []
    w_x = []
    w_y = []
    if(w < 40):
        xs = range(x, x+w)
    else:
        xs = [x+(w-1)*i/40 for i in range(41)]
    if(h < 80):
        ys = range(y, y+h)
    else:
        ys = [y+(h-1)*i/80 for i in range(81)]
    for xi in xs:
        for yi in ys:
            if(mask[yi][xi] > 0):
                w_x.append(xi)
                w_y.append(yi)
    cloud = getCoordinatesFromDepthImage(w_x, w_y)
    if(len(cloud[0])==0):
        return None
    mid = np.median(cloud, axis=1)
    stair_cloud = cloud[:, (np.abs(cloud[1]-mid[1])<0.27)&(np.abs(cloud[2]-mid[2])<0.2)&(np.abs(cloud[0]-mid[0])<1.5) ]
    height = np.percentile(stair_cloud[2], 90)
    top = stair_cloud[:, (np.abs(stair_cloud[2]-height)<0.01)&(stair_cloud[1]-np.amax(stair_cloud[1])>-0.45)]
    top = top[:, top[0]-np.amax(stair_cloud[1])>-0.1]
    if(len(top[0])==0):
        return None
    div = np.linspace(np.percentile(top[1], 20), np.percentile(top[1], 80), 11)
    edge = np.zeros((3,10))
    for i in range(len(div)-1):
        part = top[:, (top[1]>div[i])&(top[1]<div[i+1])]
        if(len(part[0])==0):
            edge[:,i]=edge[:,i-1] if i>0 else np.amax(top, axis=1)
        else:
            maxp = np.argmax(part[0])
            edge[:,i]=part[:,maxp]
    # 绘制散点图
    '''
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(stair_cloud[0], stair_cloud[1], stair_cloud[2], color='blue', s=1)
    ax.scatter(top[0], top[1], top[2], color='red', s=3)
    ax.scatter(edge[0], edge[1], edge[2], color='green', s=10)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()
    '''
    Ts = edge[0].reshape((10, 1))
    Xs = np.zeros((10, 2))
    Xs[:,0] = 1
    Xs[:,1] = edge[1].reshape((10,))
    Ws = np.linalg.inv(Xs.T.dot(Xs) ).dot(Xs.T).dot(Ts)
    theta = -atan(Ws[1])
    dist = Ws[0]*cos(theta)
    t = rospy.get_time()
    #print(degrees(theta), dist, height)
    return degrees(theta), dist, height

class UpdateThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True

    def run(self):
        rate = rospy.Rate(20)
        global edgeQ
        edgeQ = []
        rospy.loginfo("自身稳态检测线程就绪")
        while(not rospy.is_shutdown()):
            edge_line = getTopEdge()
            if(isinstance(edge_line, Iterable)):
                edgeQ.append(edge_line)
                #print(edge_line)
                if(len(edgeQ)>20):
                    edgeQ.pop(0)
                global waggle
                waggle = max((np.std(edgeQ, axis=0))/np.array((5*180/pi,edge_line[2]+0.5,edge_line[1])))
                #print('waggle=', np.std(edgeQ, axis=0)/np.array((180/pi,edge_line[2]+0.5,edge_line[1])))
            rate.sleep()
        rospy.loginfo("稳态检测线程结束")

def waitUntilStable(stable=0.03):
    while(waggle>stable):
        print('等它稳下来, waggle=', waggle)
        rospy.sleep(0.2)

def upstair(y_dir, h1=0.04, l1=0.15, h2=0.04, l2=0.15, x0=0.01, h=0.05, h_t = 0.03):
    Z_leg=[]
    X_leg=[]
    Z_torso=[]
    X_torso=[]

    step_up = int(max(75*sqrt(0.3*(h1+h2+h)), 70*sqrt(0.6*(h1+h_t)))+1)
    Z_leg+=([(h1+h2+h)*S(float(i)/step_up) for i in range(1, step_up+1)])
    Z_torso+=([(h1+h_t)*S(float(i)/step_up) for i in range(1, step_up+1)])

    step_wait = 0
    while(Z_leg[step_wait]<h1):
        step_wait+=1
    X_leg+=([0 for i in range(step_wait)])
    X_torso+=([0 for i in range(step_wait)])
    step_forward = int(75*sqrt(0.3*(l1+l2))+1)
    X_leg+=([(l1+l2)*S(float(i)/step_forward) for i in range(step_forward+1)])
    X_torso+=([(-0.01-x0)*S(float(i)/step_forward) for i in range(step_forward+1)])

    step_down = 4 #光速下降
    if(len(X_leg)-len(Z_leg)-step_down>0):
        Z_leg+=([(h1+h2+h) for i in range(len(X_leg)-len(Z_leg)-step_down)])
        Z_torso+=([(h1+h_t) for i in range(len(X_torso)-len(Z_torso)-step_down)])
    else:
        X_leg+=([(l1+l2) for i in range(len(Z_leg)+step_down-len(X_leg))])
        X_torso+=([(-0.01-x0) for i in range(len(Z_torso)+step_down-len(X_torso))])
    Z_leg+=([(h1+h2+h-h*S(float(i)/step_down)) for i in range(1, step_down+1)])
    Z_torso+=([(h1+h_t-h_t*S(float(i)/step_down)) for i in range(1, step_down+1)])
    rospy.loginfo("step num:%d", len(X_leg))
    ll_pos = getTfMat('LL')[:3,3]
    rl_pos = getTfMat('RL')[:3,3]
    waitUntilStable()
    if(y_dir==1):
        for i in range(step_wait):
            bodyMoveTo(left_leg=[(X_leg[i],0,Z_leg[i])+ll_pos], right_leg=[rl_pos], torso=[(0,0,Z_torso[i])], count=1, wait=False)
        for i in range(step_wait, len(X_leg)):
            bodyMoveTo(left_leg=[(X_leg[i],0,Z_leg[i])+ll_pos, 0], right_leg=[rl_pos], torso=[(0,0,Z_torso[i])], count=1, wait=False)
    elif(y_dir==-1):
        for i in range(step_wait):
            bodyMoveTo(left_leg=[ll_pos], right_leg=[(X_leg[i],0,Z_leg[i])+rl_pos], torso=[(0,0,Z_torso[i])], count=1, wait=False)
        for i in range(step_wait, len(X_leg)):
            bodyMoveTo(left_leg=[ll_pos], right_leg=[(X_leg[i],0,Z_leg[i])+rl_pos, 0], torso=[(0,0,Z_torso[i])], count=1, wait=False)
    waitForActionExecute()
    waitUntilStable()
    linearTrajMove(torso=[(l2,0.14*y_dir,0)])

def upstair0(y_dir, h1=0.04, l1=0.15, h2=0.04, l2=0.15, x0=0.01):
    if(y_dir==1):
        linearTrajMove(left_leg=[(0,0,h1+0.02+h2)], torso=[(0,0,h1+0.02)])
        linearTrajMove(left_leg=[(l1+l2,0,0)], torso=[(-x0-0.01,0,0)])
        linearTrajMove(left_leg=[(0,0,-0.02)], torso=[(0,0,-0.02)])
        linearTrajMove(torso=[(l2+0.02,0.14,0)])
    elif(y_dir==-1):
        linearTrajMove(right_leg=[(0,0,h1+0.02+h2)], torso=[(0,0,h1+0.02)])
        linearTrajMove(right_leg=[(l1+l2,0,0)], torso=[(-x0-0.01,0,0)])
        linearTrajMove(right_leg=[(0,0,-0.02)], torso=[(0,0,-0.02)])
        linearTrajMove(torso=[(l2+0.02,-0.14,0)])

class TrackingThread(threading.Thread):
    def __init__(self, pause, over):
        threading.Thread.__init__(self)
        self.pause = pause
        self.over = over

    def run(self):
        rate = rospy.Rate(10)
        rospy.loginfo("视觉追踪线程就绪")
        while(not rospy.is_shutdown()):
            if(self.over.is_set()):
                break
            if(not self.pause.is_set()):
                mat_pos = getCoordinateFromDepthImage(x+w/2, 479-(y+h/2))
                if(isinstance(mat_pos, Iterable)):
                    visualTrack(mat_pos)
            rate.sleep()
        rospy.loginfo("视觉追踪线程结束")

class TrackingTopThread(threading.Thread):
    def __init__(self, pause, over):
        threading.Thread.__init__(self)
        self.pause = pause
        self.over = over

    def run(self):
        rate = rospy.Rate(3)
        rospy.loginfo("视觉追踪高台后缘线程就绪")
        while(not rospy.is_shutdown()):
            if(self.over.is_set()):
                break
            if(not self.pause.is_set()):
                result = getTopEdge()
                if(isinstance(result, Iterable)):
                    dist = result[1]
                    height = result[2]
                else:
                    rate.sleep()
                    continue
                angles = ikVisualTrack([dist, 0, height])+[0, 17]
                targets[20:22] = angles[:2]
                cost0 = np.sum((measures[20:22]-targets[20:22])**2)
                rospy.sleep(0.2)
                cost = np.sum((measures[20:22]-targets[20:22])**2)
                if(cost>1 and abs(cost0-cost)<1):
                    headCtrl(angles[0], angles[1], 1)
                rate.sleep()
        rospy.loginfo("视觉追踪线程结束")

def main():
    rospy.loginfo("开始了")
    rospy.Subscriber("/sim/camera/D435/colorImage", Image, D435Colorcallback)
    setBodyhubNoStand(1)
    setArmMode(0)
    '''walkingInit()
    sendGaitCommand(0, 0, -30)
    waitForWalkingDone()
    pause = threading.Event()
    over = threading.Event()
    tracking = TrackingThread(pause, over)
    tracking.start()
    updating = UpdateThread()
    updating.start()
    starting_point = getBottomPos()
    while(not isinstance(starting_point, Iterable)):
        starting_point = getBottomPos()
        rospy.sleep(0.2)
    print("starting point:", starting_point)
    walkTheDistance(starting_point[0]-0.20,starting_point[1],0)
    #walkTheDistance(0,0,10)
    over.set()
    tracking.join()'''
    updating = UpdateThread()
    updating.start()
    pause = threading.Event()
    finish = threading.Event()
    tracking = TrackingTopThread(pause, finish)
    tracking.start()
    motionInit()
    UpdateTargets()
    linearTrajMove(torso=[(0.03,-0.07,-0.10)], left_leg=[(0,0.00,0)], right_leg=[(0,-0.00,0)])#, left_arm=[(0.21, -0.03, 0.21)], right_arm=[(0.21, 0.03, 0.21)])
    upstair(1, h1=0, l1=0, h2=0, l2=0.09)
    waitUntilStable()
    expect_pos = (0, 0.91)# theta, dist
    measure_pos = np.mean(edgeQ, axis=0)
    print(measure_pos)
    theta = measure_pos[0]-expect_pos[0]
    print('theta=', theta)
    if(abs(theta)<5):
        theta=0
    print('theta=', theta)
    bodyMoveTo(torso=[(0,0,0), theta], count=int(2*abs(theta))) #修正角度
    #bodyMoveTo(left_arm=[(0.20, 0.10, 0.05)], right_arm=[(0.20, -0.10, 0.05)])#left_arm=[(0.20, 0.15, 0.05)], right_arm=[(0.20, -0.15, 0.05)]
    l = 0.09 + measure_pos[1]-expect_pos[1]
    print("l=", l)
    t0=sim_time.time()
    upstair(-1, h1=0, l1=0.09, h2=0, l2=l+0.13*sin(radians(theta)))
    upstair(1, h1=0, l1=l)
    #1+ 0
    upstair(-1)
    #1 +2
    upstair(1)
    #2+ 3
    upstair(-1)
    #3 +4
    h1=0.04
    l1=0.15
    h2=0.00
    l2=0.00
    x0=0.01
    h=0.05
    h_t = 0.03
    Z_leg=[]
    X_leg=[]
    Z_torso=[]
    X_torso=[]

    step_up = int(max(75*sqrt(0.3*(h1+h2+h)), 70*sqrt(0.6*(h1+h_t)))+1)
    Z_leg+=([(h1+h2+h)*S(float(i)/step_up) for i in range(1, step_up+1)])
    Z_torso+=([(h1+h_t)*S(float(i)/step_up) for i in range(1, step_up+1)])

    step_wait = 0
    while(Z_leg[step_wait]<h1):
        step_wait+=1
    X_leg+=([0 for i in range(step_wait)])
    X_torso+=([0 for i in range(step_wait)])
    step_forward = int(75*sqrt(0.3*(l1+l2))+1)
    X_leg+=([(l1+l2)*S(float(i)/step_forward) for i in range(step_forward+1)])
    X_torso+=([(-0.01-x0)*S(float(i)/step_forward) for i in range(step_forward+1)])

    step_down = 4 #光速下降
    if(len(X_leg)-len(Z_leg)-step_down>0):
        Z_leg+=([(h1+h2+h) for i in range(len(X_leg)-len(Z_leg)-step_down)])
        Z_torso+=([(h1+h_t) for i in range(len(X_torso)-len(Z_torso)-step_down)])
    else:
        X_leg+=([(l1+l2) for i in range(len(Z_leg)+step_down-len(X_leg))])
        X_torso+=([(-0.01-x0) for i in range(len(Z_torso)+step_down-len(X_torso))])
    Z_leg+=([(h1+h2+h-h*S(float(i)/step_down)) for i in range(1, step_down+1)])
    Z_torso+=([(h1+h_t-h_t*S(float(i)/step_down)) for i in range(1, step_down+1)])
    rospy.loginfo("step num:%d", len(X_leg))
    ll_pos = getTfMat('LL')[:3,3]
    rl_pos = getTfMat('RL')[:3,3]
    waitUntilStable()
    for i in range(step_wait):
        bodyMoveTo(left_leg=[(X_leg[i],0,Z_leg[i])+ll_pos], right_leg=[rl_pos], torso=[(0,0,Z_torso[i])], count=1, wait=False)
    for i in range(step_wait, len(X_leg)):
        bodyMoveTo(left_leg=[(X_leg[i],0,Z_leg[i])+ll_pos], right_leg=[rl_pos], torso=[(0,0,Z_torso[i])], count=1, wait=False)
    waitForActionExecute()
    #4+ 4
    waitUntilStable()
    linearTrajMove(torso=[(-0.02,0.07,0.10)], left_leg=[(0,-0.00,0),0], right_leg=[(0,0.00,0),0])
    t=sim_time.time()
    print('耗时%f秒' % (t-t0))
    rospy.sleep(1)
    print('到了')
    finish.set()

    



if __name__ == '__main__':
    
    def rosShutdownHook():
        setBodyhubNoStand(0)
        setArmMode(1)
        resetBodyhub()
        rospy.signal_shutdown('node_close')

    rospy.init_node('task10_upstair')
    rospy.on_shutdown(rosShutdownHook)
    rospy.sleep(0.5)
    UpdateTargets() 
    main()