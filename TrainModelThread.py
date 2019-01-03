# -*- coding: utf-8 -*-
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
import os
import time

os.environ['PYSPARK_PYTHON'] = '/home/sunbite/anaconda3/bin/python'


class TrainModelThread(QThread):
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
            self.trainModel(self.sendlog)
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
        # 删除协同训练保存的model文件
        rm_30model_stdin, rm_30model_stdout, rm_30model_stderr = self.ssh.exec_command(
            'rm /home/sunbite/MFSSEL/model/30.model')
        print(rm_30model_stdout.read())
        print(rm_30model_stderr.read())
        # 删除协同训练保存的model文件
        rm_81model_stdin, rm_81model_stdout, rm_81model_stderr = self.ssh.exec_command(
            'rm /home/sunbite/MFSSEL/model/81.model')
        print(rm_81model_stdout.read())
        print(rm_81model_stderr.read())
        # 删除协同训练保存的model文件
        rm_hogmodel_stdin, rm_hogmodel_stdout, rm_hogmodel_stderr = self.ssh.exec_command(
            'rm /home/sunbite/MFSSEL/model/hog.model')
        print(rm_hogmodel_stdout.read())
        print(rm_hogmodel_stderr.read())

    def trainModel(self, sendlog):
        """
        模型训练
        :param sendlog:要发送的log信号
        :return:
        """
        # self.deleteAndCreateTmp()
        # # 协同训练保存model
        # # trainandpredictonspark_stdin, trainandpredictonspark_stdout, trainandpredictonspark_stderr = self.ssh.exec_command(
        # #     'export PATH=/home/sunbite/anaconda3/bin:$PATH;/home/sunbite/spark-2.1.2/bin/spark-submit --py-files /home/sunbite/PycharmProjects/myFirstPoint/ListParam.py,/home/sunbite/PycharmProjects/myFirstPoint/Co_KNN_SVM_New.py,/home/sunbite/PycharmProjects/myFirstPoint/Co_KNN_SVM_Utilities.py --master local[2] /home/sunbite/PycharmProjects/myFirstPoint/CoKNNSVMTrainAndPredictOnSpark.py')
        # trainandpredictonspark_stdin, trainandpredictonspark_stdout, trainandpredictonspark_stderr = self.ssh.exec_command(
        #     'export PATH=/home/sunbite/anaconda3/bin:$PATH;python /home/sunbite/PycharmProjects/myFirstPoint/MFSSEL.py')
        # print(repr(trainandpredictonspark_stdout.read().decode()))

        time.sleep(10)
        sendlog.emit("模型训练完成。")
