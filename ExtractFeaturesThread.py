# -*- coding: utf-8 -*-
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
import os
import time
os.environ['PYSPARK_PYTHON']='/home/sunbite/anaconda3/bin/python'

class ExtractFeaturesThread(QThread):
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
            self.extractFeatures(self.sendlog)
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
        # 删除特征提取准备的参数文件
        rm_keyframepath_stdin, rm_keyframepath_stdout, rm_keyframepath_stderr = self.ssh.exec_command(
            'rm /home/sunbite/MFSSEL/keyframepath.txt')
        print(rm_keyframepath_stdout.read())
        print(rm_keyframepath_stderr.read())
        # 删除特征提取文件夹
        rm_features_stdin, rm_features_stdout, rm_features_stderr = self.ssh.exec_command(
            'rm -r /home/sunbite/MFSSEL/features')
        print(rm_features_stdout.read())
        print(rm_features_stderr.read())

    def extractFeatures(self, sendlog):
        """
        模型训练
        :param sendlog:要发送的log信号
        :return:
        """

        time.sleep(10)
        #self.deleteAndCreateTmp()
        # 特征提取参数准备
        # keyframepathwriter_stdin, keyframepathwriter_stdout, keyframepathwriter_stderr = self.ssh.exec_command(
        #     'export PATH=/home/sunbite/anaconda3/bin:$PATH;python /home/sunbite/PycharmProjects/myFirstPoint/KeyFramePathWriter.py')
        # print(keyframepathwriter_stdout.read())
        # # 特征提取
        # featuresextractoronspark_stdin, featuresextractoronspark_stdout, featuresextractoronspark_stderr = self.ssh.exec_command(
        #     'export PATH=/home/sunbite/anaconda3/bin:$PATH;/home/sunbite/spark-2.1.2/bin/spark-submit --py-files /home/sunbite/PycharmProjects/myFirstPoint/GetFeatures.py,/home/sunbite/PycharmProjects/myFirstPoint/FeaturesExtractor.py --master local[2] /home/sunbite/PycharmProjects/myFirstPoint/FeaturesExtractorOnSpark.py')
        # print(featuresextractoronspark_stdout.read())

        sendlog.emit("特征提取完成。")
