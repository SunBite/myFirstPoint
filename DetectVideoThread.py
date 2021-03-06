# -*- coding: utf-8 -*-
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
import time
import re
import paramiko
import VideoDetector
import os


class DetectVideoThread(QThread):
    # 线程信号
    sendlog = pyqtSignal(str)

    def __init__(self, func, videoList):
        super().__init__()
        self.stopped = True
        self.mutex = QMutex()
        self.func = func
        self.__videoList = videoList
        self.__classNameMap = {1: "basketball", 2: "biking", 3: "diving", 4: "golf_swing", 5: "horse_riding",
                               6: "soccer_juggling", 7: "swing", 8: "tennis_swing", 9: "trampoline_jumping",
                               10: "volleyball_spiking",
                               11: "walking"}

    def run(self):
        """
        线程开始运行
        :return:
        """
        # 互斥锁
        with QMutexLocker(self.mutex):
            self.stopped = False
        if not self.stopped:
            self.sendlog.connect(self.func)
            self.detectVideo(self.sendlog)
        else:
            return

    def is_stopped(self):
        """
        线程状态是否是停止
        :return:
        """
        with QMutexLocker(self.mutex):
            return self.stopped

    def stop(self):
        """
        线程停止
        :return:
        """
        # 互斥锁
        with QMutexLocker(self.mutex):
            self.stopped = True

    def detectVideo(self, sendlog):
        """
        检测视频
        :param sendlog:要发送的log信号
        :return:
        """
        log = ''
        rightNum = 0
        for i in range(0, len(self.__videoList)):
            vd = VideoDetector.VideoDetector(self.__videoList[i], "/home/sunbite/MFSSEL/model")
            predictLabel, trueLabel = vd.getLabel()
            if predictLabel == trueLabel:
                rightNum = rightNum + 1
            predictClassName = self.__classNameMap[predictLabel]
            trueClassName = self.__classNameMap[trueLabel]
            log += '视频%s：\n检测类别为：%s。\n实际类别为：%s。 \n' % (
                os.path.basename(self.__videoList[i]), predictClassName, trueClassName)
        log += "此次检测准确率为:%s。\n" % (rightNum / len(self.__videoList))
        log += "视频检测完成。"
        sendlog.emit(log)
