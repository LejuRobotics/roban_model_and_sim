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


def main():
    rospy.loginfo("开始了")

    mat_pos = [0.17906035, -0.03357439, 0.00390709]

    setBodyhubNoStand(1)
    setArmMode(0)
    # motionInit()
    walkingInit()

    cup_in_hand = (-0.07, -0.05, -0.016)  # 很重要
    arm_prepare = [0.15, -0.15, 0.06]

    rospy.loginfo("伸手")
    bodyMoveTo(
        right_arm=[arm_prepare, cup_in_hand], count=10)

    rospy.loginfo("放杯子")
    linearTrajMoveTo(
        right_arm=[mat_pos, cup_in_hand], divide=10)
    rospy.loginfo("松手")
    gripperCtrl(right=45)
    rospy.loginfo("抽手")
    linearTrajMove(right_arm=[(0, 0, 0.06), cup_in_hand], divide=10)
    linearTrajMove(right_arm=[(-0.03, -0.04, 0), cup_in_hand], divide=10)
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
