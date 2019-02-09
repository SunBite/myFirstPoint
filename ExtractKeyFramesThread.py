# -*- coding: utf-8 -*-
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
import os
import time
os.environ['PYSPARK_PYTHON']='/home/sunbite/anaconda3/bin/python'

class ExtractKeyFramesThread(QThread):
    # 线程信号
    sendlog = pyqtSignal(str)

    def __init__(self, func, ssh):
        super().__init__()
        self.stopped = True
        self.mutex = QMutex()
        self.func = func
        self.ssh = ssh

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
            self.extractKeyFrame(self.sendlog)
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

    def deleteAndCreateTmp(self):
        """
        删除临时文件
        :return:
        """
        # 删除关键帧文件夹
        rm_keyframe_stdin, rm_keyframe_stdout, rm_keyframe_stderr = self.ssh.exec_command(
            'rm -r /home/sunbite/MFSSEL/keyframe_on_spark')
        print(rm_keyframe_stdout.read())
        print(rm_keyframe_stderr.read())
        if rm_keyframe_stderr:
            # 创建空的关键帧文件夹
            mk_keyframe_stdin, mk_keyframe_stdout, mk_keyframe_stderr = self.ssh.exec_command(
                'mkdir /home/sunbite/MFSSEL/keyframe_on_spark')
            print(mk_keyframe_stdout.read())
            print(mk_keyframe_stderr.read())
        # 删除关键帧提取准备的参数文件
        rm_videopath_stdin, rm_videopath_stdout, rm_videopath_stderr = self.ssh.exec_command(
            'rm /home/sunbite/MFSSEL/videopath.txt')
        print(rm_videopath_stdout.read())
        print(rm_videopath_stderr.read())

    def extractKeyFrame(self, sendlog):
        """
        模型训练
        :param sendlog:要发送的log信号
        :return:
        """
        self.deleteAndCreateTmp()
        # 关键帧提取参数准备
        videopathwriter_stdin, videopathwriter_stdout, videopathwriter_stderr = self.ssh.exec_command(
            'export PATH=/home/sunbite/anaconda3/bin:$PATH;python /home/sunbite/PycharmProjects/myFirstPoint/VideoPathWriter.py')
        print(videopathwriter_stdout.read())
        # 关键帧提取
        keyframeextractoronspark_stdin, keyframeextractoronspark_stdout, keyframeextractoronspark_stderr = self.ssh.exec_command(
            'export PATH=/home/sunbite/anaconda3/bin:$PATH;/home/sunbite/spark-2.1.2/bin/spark-submit --py-files /home/sunbite/PycharmProjects/myFirstPoint/KeyFrameExtractor.py --master local[2] /home/sunbite/PycharmProjects/myFirstPoint/KeyFrameExtractorOnSpark.py')
        print(keyframeextractoronspark_stdout.read())

        sendlog.emit("关键帧提取完成。")
