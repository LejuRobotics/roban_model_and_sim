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
    setBodyhubNoStand(1)
    setArmMode(0)
    # motionInit()
    walkingInit()
    handle_pos = [0.17468236, - 0.14123561, 0.04488388]
    arm_prepare = [0.15, -0.2, 0.1]
    gripperCtrl(right=45)
    bodyMoveTo(right_arm=[arm_prepare, "BT"], count=10)

    rospy.loginfo("伸手")
    linearTrajMoveTo(
        right_arm=[handle_pos + np.array([0, 0, 0]), "BT"], divide=10)
    rospy.loginfo("夹住")
    gripperCtrl(right=10)
    rospy.loginfo("关闭")
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
