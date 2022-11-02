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
    motionInit()
    button_pos = (0.21810245, -0.09728054, 0.10072714)
    UpdateTargets()
    arm_prepare = [0.15, -0.15, 0.1]
    bodyMoveTo(right_arm=[arm_prepare], count=10)
    linearTrajMoveTo(right_arm=[button_pos+np.array([0, 0, 0])], divide=10)
    linearTrajMoveTo(right_arm=[button_pos+np.array([-0.03, 0, 0])], divide=10)
    # raw_input("完成了...")
    # print("finish")


if __name__ == '__main__':

    def rosShutdownHook():
        setBodyhubNoStand(0)
        setArmMode(1)
        resetBodyhub()
        rospy.signal_shutdown('node_close')

    rospy.init_node('task02_press_button', anonymous=True)
    rospy.on_shutdown(rosShutdownHook)
    main()
