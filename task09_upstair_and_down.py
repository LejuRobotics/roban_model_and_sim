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

import numpy as np
import math

from collections import Iterable

'''
看起来楼梯一格的高度是4cm，长度是15cm
'''

def upstair(y_dir, h1=0.04, l1=0.15, h2=0.04, l2=0.15, x0=0.01):
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

def downstair(y_dir, h1=0.04, l1=0.15, h2=0.04, l2=0.15, x0=0.01):
    if(y_dir==1):
        linearTrajMove(torso=[(l1+0.01+x0,0.14,0)])
        linearTrajMove(right_leg=[(0,0,0.02)], torso=[(0,0,0.02)])
        linearTrajMove(right_leg=[(l1+l2,0,0)], torso=[(-0.02,0,0)])
        linearTrajMove(right_leg=[(0,0,-(h1+h2+0.02) )], torso=[(0,0,-h2-0.02)])
    elif(y_dir==-1):
        linearTrajMove(torso=[(l1+0.01+x0,-0.14,0)])
        linearTrajMove(left_leg=[(0,0,0.02)], torso=[(0,0,0.02)])
        linearTrajMove(left_leg=[(l1+l2,0,0)], torso=[(-0.02,0,0)])
        linearTrajMove(left_leg=[(0,0,-(h2+h1+0.02) )], torso=[(0,0,-h2-0.02)])

def main():
    rospy.loginfo("开始了")
    setBodyhubNoStand(1)
    setArmMode(0)
    motionInit()
    linearTrajMove(torso=[(0,0.07,-0.10)])
    waitForActionExecute()
    upstair(-1, h1=0, l1=0, h2=0, l2=0.10, x0=0)
    #-
    #  -
    upstair(1, h1=0, l1=0.10, h2=0, l2=0.09)
    #    -
    #  -
    upstair(-1, h1=0, l1=0.09, h2=0, l2=0)
    #    -
    #    -
    linearTrajMove(torso=[(0.02,0,0)])
    upstair(1, h1=0, l1=0)
    #1+ 0
    upstair(-1)
    #1 +2
    upstair(1)
    #2+ 3
    upstair(-1)
    #3 +4
    #linearTrajMove(torso=[(-0.01,0,0)])
    upstair(1, h2=0, l2=0.10)
    #4+ 4
    
    linearTrajMove(torso=[(-0.02,0,0)])

    linearTrajMove(right_leg=[(0,0,0.02)], torso=[(0,0,0.02)])
    linearTrajMove(right_leg=[(0.10,0,0)], torso=[(-0.01,0,0)])
    linearTrajMove(right_leg=[(0,0,-0.02)], torso=[(0,0,-0.02)])
    #4+ 4
    linearTrajMove(torso=[(-0.02,0,0)])

    downstair(-1, h1=0, l1=0.11)
    #3 +4
    downstair(1)
    #3+ 2
    downstair(-1)
    #1 +2
    downstair(1)
    #1+ 0
    downstair(-1, h2=0, l2=0)

    linearTrajMove(torso=[(0.03,0,0)])

    linearTrajMove(torso=[(0,-0.07,0.10)])

if __name__ == '__main__':
    
    def rosShutdownHook():
        rospy.loginfo('主程序运行结束')
        setBodyhubNoStand(0)
        setArmMode(1)
        resetBodyhub()
        rospy.signal_shutdown('node_close')

    rospy.init_node('task09_practice', anonymous=True)
    rospy.sleep(0.5)
    UpdateTargets()
    rospy.on_shutdown(rosShutdownHook)    
    main()
