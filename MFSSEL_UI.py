# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QMainWindow, QApplication, QDesktopWidget, qApp, QFileDialog, QMessageBox, QAction, \
    QPushButton
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import QRect, QTimer, QDir
from PyQt5.Qt import QLineEdit
import sys
import os
import cv2
from TrainModelThread import TrainModelThread
from ExtractFeaturesThread import ExtractFeaturesThread
from ExtractKeyFramesThread import ExtractKeyFramesThread
from DetectVideoThread import DetectVideoThread
import paramiko


class MFSSEL_UI(QMainWindow):

    def __init__(self):
        super().__init__()
        # 初始化UI
        self.initUI()
        # 创建action
        self.initAction()
        self.center()
        self.setWindowTitle("基于Spark的视频语义检测系统")
        self.setFixedSize(750, 600)
        self.__videoList = []

    def initUI(self):
        """
        初始化UI
        :return:
        """
        # QAction是一个抽象类，可以通过菜单栏，工具栏，或者快捷键实现，
        # 以下代码实现了带有图标和Exit的菜单的创建，同时设置了执行命令的快捷键以及鼠标悬停在这个菜单上的提示信息
        exitAct = QAction(QIcon('exit.ico'), 'Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Press and quit')

        # 执行此动作时，发射一个triggered信号，这个信号和quit连接，执行相关事件
        exitAct.triggered.connect(qApp.quit)

        # 状态栏
        self.statusBar()
        # tbar = self.addToolBar('Exit')
        # tbar.addAction(exitAct)

        # 服务器地址label
        self.label_serverAddress = QtWidgets.QLabel(self)
        self.label_serverAddress.setText("服务器地址:")
        self.label_serverAddress.resize(120, 30)
        self.label_serverAddress.move(30, 35)
        # 服务器地址
        self.textbox_serverAddress = QLineEdit(self)
        self.textbox_serverAddress.move(115, 35)

        # 用户名label
        self.label_userName = QtWidgets.QLabel(self)
        self.label_userName.setText("用户名:")
        # self.label_userName.resize(120, 30)
        self.label_userName.move(220, 35)
        # 用户名LineEdit
        self.textbox_userName = QLineEdit(self)
        self.textbox_userName.move(275, 35)

        # 密码label
        self.label_password = QtWidgets.QLabel(self)
        self.label_password.setText("密码:")
        # self.label_userName.resize(120, 30)
        self.label_password.move(380, 35)
        # 密码LineEdit
        self.textbox_password = QLineEdit(self)
        self.textbox_password.move(420, 35)
        self.textbox_password.setEchoMode(QLineEdit.Password)

        # 链接服务器按钮
        self.bt_connectServer = QPushButton('连接服服务器', self)
        self.bt_connectServer.clicked.connect(self.on_click)
        self.bt_connectServer.move(620, 35)
        # 菜单栏
        self.menubar = self.menuBar()

        # 打开menu
        self.menu_openvideos = QtWidgets.QMenu()
        self.menu_openvideos.setTitle("添加")
        self.menubar.addMenu(self.menu_openvideos)

        # 处理menu
        self.menu_dispose = QtWidgets.QMenu()
        self.menu_dispose.setTitle("处理")
        self.menubar.addMenu(self.menu_dispose)

        # 帮助menu
        self.menu_help = QtWidgets.QMenu()
        self.menu_help.setTitle("帮助")
        self.menubar.addMenu(self.menu_help)

        # 检测视频文件目录label
        self.label_videofiledir = QtWidgets.QLabel(self)
        self.label_videofiledir.setText("待检测视频列表:")
        self.label_videofiledir.resize(120, 30)
        self.label_videofiledir.move(380, 60)

        # 检测视频文件目录
        self.treeview_videofiledir = QtWidgets.QTreeView(self)
        self.treeview_videofiledir.resize(340, 480)
        self.treeview_videofiledir.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.treeview_videofiledir.setSizeAdjustPolicy(QtWidgets.QTreeView.AdjustToContents)
        self.treeview_videofiledir.header().setFixedWidth(700)
        self.model = QtGui.QStandardItemModel()
        self.model.setHorizontalHeaderItem(0, QtGui.QStandardItem("视频列表"))
        self.treeview_videofiledir.setModel(self.model)
        self.treeview_videofiledir.move(380, 90)

        # 播放视频的线程
        # self.playvideothread = pvt.PlayVideoThread()

        # 双击文件目录连接启动播放视频的线程
        self.treeview_videofiledir.doubleClicked.connect(self.startPlayVideoThread)

        # 视频播放label
        self.label_videoplay = QtWidgets.QLabel(self)
        self.label_videoplay.setText("视频播放：")
        self.label_videoplay.move(30, 60)

        # 显示视频框
        self.picturelabel = QtWidgets.QLabel(self)

        init_image = QPixmap(QDir.currentPath()+"/no_video.jpg").scaled(320, 240)
        self.picturelabel.setPixmap(init_image)
        self.picturelabel.resize(320, 240)
        self.picturelabel.move(30, 90)

        # 显示处理信息label
        self.label_showmessage = QtWidgets.QLabel(self)
        self.label_showmessage.setText("处理信息：")
        self.label_showmessage.move(30, 340)
        # 显示处理信息框
        self.textedit = QtWidgets.QTextEdit(self)
        self.textedit.setReadOnly(True)
        self.textedit.resize(320, 200)
        self.textedit.move(30, 370)

    def initAction(self):
        """
        创建action
        :return:
        """
        # 选取待检测视频menu的Action
        self.action_openvideos_choosevideos = QtWidgets.QAction("添加视频到待检测视频列表", self)
        self.action_openvideos_choosevideos.triggered.connect(self.chooseVideos)
        self.action_openvideos_choosefolder = QtWidgets.QAction("添加文件夹到待检测视频列表", self)
        self.action_openvideos_choosefolder.triggered.connect(self.chooseFolder)
        self.action_openvideos_clear = QtWidgets.QAction("清空待检测视频列表", self)
        self.action_openvideos_clear.triggered.connect(self.openVideos_clear)
        self.menu_openvideos.addAction(self.action_openvideos_choosevideos)
        self.menu_openvideos.addAction(self.action_openvideos_choosefolder)
        self.menu_openvideos.addAction(self.action_openvideos_clear)
        self.menu_openvideos.addSeparator()
        # 处理menu的Action
        self.action_keyframe = QtWidgets.QAction("Spark下关键帧提取", self)
        self.action_keyframe.setEnabled(False)
        self.action_keyframe.triggered.connect(self.extractKeyFrames)
        self.action_feature = QtWidgets.QAction("Spark下特征提取", self)
        self.action_feature.setEnabled(False)
        self.action_feature.triggered.connect(self.extractFeatures)
        self.action_trainmodel = QtWidgets.QAction("Spark下训练模型", self)
        self.action_trainmodel.setEnabled(False)
        self.action_trainmodel.triggered.connect(self.trainModel)
        self.action_detect = QtWidgets.QAction("检测视频", self)
        self.action_detect.triggered.connect(self.detectVideo)
        self.menu_dispose.addAction(self.action_keyframe)
        self.menu_dispose.addAction(self.action_feature)
        self.menu_dispose.addAction(self.action_trainmodel)
        self.menu_dispose.addAction(self.action_detect)
        self.menu_dispose.addSeparator()
        # 帮助menu的Action
        self.action_lookuphelp = QtWidgets.QAction("帮助", self)
        self.action_lookuphelp.triggered.connect(self.lookuphelpfile)
        self.action_about = QtWidgets.QAction("关于", self)
        self.action_about.triggered.connect(self.lookupaboutfile)
        self.menu_help.addAction(self.action_lookuphelp)
        self.menu_help.addAction(self.action_about)
        self.menu_help.addSeparator()

    def center(self):
        """
        控制窗口显示在屏幕中心的方法
        :return:
        """
        # 获得窗口
        qr = self.frameGeometry()
        # 获得屏幕中心点
        cp = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        qr.moveCenter(cp)
        # self.move(qr.topLeft())
        self.move(320, 70)

    def on_click(self):
        """
        链接服务器按钮点击
        :return:
        """
        textbox_serverAddress_Value = self.textbox_serverAddress.text().strip()
        textbox_userName_Value = self.textbox_userName.text().strip()
        textbox_password_Value = self.textbox_password.text().strip()
        if (textbox_serverAddress_Value is "") or len(textbox_serverAddress_Value) == 0:
            QMessageBox.information(self, "注意", "服务器地址为空！")
            self.textbox_serverAddress.setFocus()
        elif (textbox_userName_Value is "") or len(textbox_userName_Value) == 0:
            QMessageBox.information(self, "注意", "用户名为空！")
            self.textbox_userName.setFocus()
        elif (textbox_password_Value is "") or len(textbox_password_Value) == 0:
            QMessageBox.information(self, "注意", "密码为空！")
            self.textbox_password.setFocus()
        try:
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(hostname=textbox_serverAddress_Value, username=textbox_userName_Value,
                             password=textbox_password_Value)
            # self.ssh.connect(hostname=hostname, username=username, password=password)
            stdin, stdout, stderr = self.ssh.exec_command('who')
            whoMess = repr(stdout.read().decode())
            search = whoMess.find(textbox_userName_Value)
            if search:
                info = '服务器连接成功。'
                self.flag = 1
                self.action_trainmodel.setEnabled(True)
                self.action_keyframe.setEnabled(True)
                self.action_feature.setEnabled(True)
            else:
                info = '服务器连接失败。'
                self.flag = 2
                self.action_trainmodel.setEnabled(False)
                self.action_keyframe.setEnabled(False)
                self.action_feature.setEnabled(False)

        except:
            info = '服务器连接失败。'
            self.flag = 2
            self.action_trainmodel.setEnabled(False)
            self.action_keyframe.setEnabled(False)
            self.action_feature.setEnabled(False)
        self.textedit.append(info)

    def chooseVideos(self):
        """
        选取要检测的视频加入到视频列表中
        :return:
        """
        files, filetype = QFileDialog.getOpenFileNames(None, "视频选择", os.path.expanduser("~"),
                                                       "Video Files (*.avi *.mpg *.mp4)")
        if files:
            # 清空model
            self.openVideos_clear()
            self.__videoList = files
            videoItem = QtGui.QStandardItem(os.path.dirname(files[0]))
            videoItem.setEditable(False)
            for file in files:
                videoItemChild = QtGui.QStandardItem(os.path.basename(file))
                videoItemChild.setEditable(False)
                videoItem.appendRow(videoItemChild)
            self.model.appendRow(videoItem)
            self.treeview_videofiledir.setModel(self.model)

    def chooseFolder(self):
        """
        把文件夹中的视频加入视频列表中
        :return:
        """
        directory = QFileDialog.getExistingDirectory(None, "文件夹选择", os.path.expanduser("~"))
        if os.path.isdir(directory):
            # 清空model
            self.openVideos_clear()
            # 遍历文件夹
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filePath = os.path.join(dirpath, filename)
                    self.__videoList.append(filePath)
                if filenames:
                    videoItem = QtGui.QStandardItem(dirpath)
                    videoItem.setEditable(False)
                    for filename in filenames:
                        # 获取文件名后缀，通过os.path.splitext（path）分离文件名和扩展名
                        ext = os.path.splitext(filename)[1]
                        # 将文件名统一转化为小写
                        ext = ext.lower()
                        if ext == '.avi' or ext == '.mpg' or ext == '.mp4':
                            videoItemChild = QtGui.QStandardItem(os.path.basename(filename))
                            videoItemChild.setEditable(False)
                            videoItem.appendRow(videoItemChild)
                    self.model.appendRow(videoItem)
                    self.treeview_videofiledir.setModel(self.model)

    def openVideos_clear(self):
        """
        清空视频列表
        :return:
        """
        self.__videoList.clear()
        self.model.clear()
        headerItem = QtGui.QStandardItem("视频列表")
        headerItem.setEditable(False)
        self.model.setHorizontalHeaderItem(0, headerItem)
        self.treeview_videofiledir.setModel(self.model)
        init_image = QPixmap(QDir.currentPath()+"/no_video.jpg").scaled(320, 240)
        self.picturelabel.setPixmap(init_image)

    def startPlayVideoThread(self):
        """
        启动播放视频的线程
        :return:
        """
        # 获取选中的当前的Index
        index = self.treeview_videofiledir.selectionModel().currentIndex()
        # 如果当前Index有父节点，也就是说点击视频文件名
        if index.parent().data():
            # 视频的路径
            videopath = index.parent().data() + os.sep + index.data()
            self.playCapture = cv2.VideoCapture(videopath)
            fps = self.playCapture.get(cv2.CAP_PROP_FPS)
            self.timer = QTimer()
            self.timer.timeout.connect(self.playVideo)
            self.timer.start(1000 / fps)
            # self.playvideothread.setFps(fps)
            # # 连接播放视频槽
            # self.playvideothread.signal.connect(self.playVideo)
            # self.playvideothread.start()

    def playVideo(self):
        """
        播放视频
        :return:
        """
        if self.playCapture.isOpened():
            ret, frame = self.playCapture.read()
            if ret:
                # self.treeview_videofiledir.setDisabled(True)
                # 获取视频播放label的大小
                s = self.picturelabel.rect()
                # frame重置大小
                R_frame = cv2.resize(frame, (QRect.width(s), QRect.height(s)))
                if R_frame.ndim == 3:
                    R_frame_RGB = cv2.cvtColor(R_frame, cv2.COLOR_BGR2RGB)
                elif R_frame.ndim == 2:
                    R_frame_RGB = cv2.cvtColor(R_frame, cv2.COLOR_GRAY2BGR)
                qImage = QtGui.QImage(R_frame_RGB[:], R_frame_RGB.shape[1], R_frame_RGB.shape[0],
                                      QtGui.QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qImage)
                self.picturelabel.setPixmap(pixmap)
            else:
                # 释放VideoCapture
                self.playCapture.release()
                # 关闭线程
                self.timer.stop()
                # self.playvideothread.stop()
                # self.treeview_videofiledir.setDisabled(False)

    def extractKeyFrames(self):
        """
        关键帧提取
        :return:
        """
        if os.path.exists("/home/sunbite/MFSSEL/keyframe"):
            reply = QMessageBox.information(self,
                                            "注意",
                                            "已有关键帧信息，是否重新提取关键帧？",
                                            QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.startExtractKeyFramesThread()

            else:
                pass
        else:
            self.startExtractKeyFramesThread()

    def startExtractKeyFramesThread(self):
        """
        调用关键帧提取线程
        :return:
        """
        self.textedit.append("正在提取关键帧...")
        self.action_keyframe.setEnabled(False)
        self.action_feature.setEnabled(False)
        self.action_trainmodel.setEnabled(False)
        self.action_detect.setEnabled(False)
        self.extractKeyFramesThread = ExtractKeyFramesThread(self.receiveLogForExtractKeyFrames, self.ssh)
        # self.trainmodelthread.signal.connect(self.trainModel)
        self.extractKeyFramesThread.start()

    def receiveLogForExtractKeyFrames(self, msg):
        """
        接受log信息
        :param msg: log信息
        :return:
        """
        self.textedit.append(msg)
        self.extractKeyFramesThread.stop()
        self.action_keyframe.setEnabled(True)
        self.action_feature.setEnabled(True)
        self.action_trainmodel.setEnabled(True)
        self.action_detect.setEnabled(True)

    def extractFeatures(self):
        """
        特征提取
        :return:
        """
        if os.path.exists("/home/sunbite/MFSSEL/features"):
            reply = QMessageBox.information(self,
                                            "注意",
                                            "已有特征信息，是否重新提取特征？",
                                            QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                if os.path.exists("/home/sunbite/MFSSEL/keyframe"):
                    self.startExtractFeaturesThread()
                else:
                    QMessageBox.information(self,
                                            "注意",
                                            "关键帧信息不存在，请重新提取关键帧！")

            else:
                pass
        else:
            if os.path.exists("/home/sunbite/MFSSEL/keyframe"):
                self.startExtractFeaturesThread()
            else:
                QMessageBox.information(self,
                                        "注意",
                                        "关键帧信息不存在，请重新提取关键帧！")

    def startExtractFeaturesThread(self):
        """
        调用训练model线程
        :return:
        """
        self.textedit.append("正在提取特征...")
        self.action_keyframe.setEnabled(False)
        self.action_feature.setEnabled(False)
        self.action_trainmodel.setEnabled(False)
        self.action_detect.setEnabled(False)
        self.extractfeaturesThread = ExtractFeaturesThread(self.receiveLogForExtractFeatures, self.ssh)
        # self.trainmodelthread.signal.connect(self.trainModel)
        self.extractfeaturesThread.start()

    def receiveLogForExtractFeatures(self, msg):
        """
        接受log信息
        :param msg: log信息
        :return:
        """
        self.textedit.append(msg)
        self.extractfeaturesThread.stop()
        self.action_keyframe.setEnabled(True)
        self.action_feature.setEnabled(True)
        self.action_trainmodel.setEnabled(True)
        self.action_detect.setEnabled(True)

    def trainModel(self):
        """
        训练model
        :return:
        """
        if os.path.exists("/home/sunbite/MFSSEL/model"):
            reply = QMessageBox.information(self,
                                            "注意",
                                            "已有训练模型，是否重新训练模型？",
                                            QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                if os.path.exists("/home/sunbite/MFSSEL/keyframe"):
                    if os.path.exists("/home/sunbite/MFSSEL/features"):
                        self.startTrainModelThread()
                    else:
                        QMessageBox.information(self,
                                                "注意",
                                                "特征信息不存在，请重新提取特征！")
                else:
                    QMessageBox.information(self,
                                            "注意",
                                            "关键帧信息不存在，请重新提取关键帧！")

            else:
                pass
        else:
            if os.path.exists("/home/sunbite/MFSSEL/keyframe"):
                if os.path.exists("/home/sunbite/MFSSEL/features"):
                    self.startTrainModelThread()
                else:
                    QMessageBox.information(self,
                                            "注意",
                                            "特征信息不存在，请重新提取特征！")
            else:
                QMessageBox.information(self,
                                        "注意",
                                        "关键帧信息不存在，请重新提取关键帧！")

    def startTrainModelThread(self):
        """
        调用训练model线程
        :return:
        """
        self.textedit.append("正在训练模型...")
        self.action_keyframe.setEnabled(False)
        self.action_feature.setEnabled(False)
        self.action_trainmodel.setEnabled(False)
        self.action_detect.setEnabled(False)
        self.trainmodelthread = TrainModelThread(self.receiveLogForTrainModel, self.ssh)
        # self.trainmodelthread.signal.connect(self.trainModel)
        self.trainmodelthread.start()

    def receiveLogForTrainModel(self, msg):
        """
        接受log信息
        :param msg: log信息
        :return:
        """
        self.textedit.append(msg)
        self.trainmodelthread.stop()
        self.action_keyframe.setEnabled(True)
        self.action_feature.setEnabled(True)
        self.action_trainmodel.setEnabled(True)
        self.action_detect.setEnabled(True)

    def detectVideo(self):
        """
        检测视频
        :return:
        """
        print(len(self.__videoList))
        if len(self.__videoList) == 0:
            reply = QMessageBox.information(self,
                                            "注意",
                                            "待检测视频为空，请添加待检测视频！",
                                            QMessageBox.Ok)
        else:
            if os.path.exists("/home/sunbite/MFSSEL/model"):
                self.startDetectVideoThread()
            else:
                QMessageBox.information(self,
                                        "注意",
                                        "模型不存在，请重新训练模型！")

    def startDetectVideoThread(self):
        """
        调用检测线程
        :return:
        """
        self.textedit.append("正在检测视频...")
        self.action_openvideos_choosevideos.setEnabled(False)
        self.action_openvideos_choosefolder.setEnabled(False)
        self.action_openvideos_clear.setEnabled(False)
        self.action_keyframe.setEnabled(False)
        self.action_feature.setEnabled(False)
        self.action_trainmodel.setEnabled(False)
        self.action_detect.setEnabled(False)
        self.detectvideothread = DetectVideoThread(self.receiveLogForDetectVideo, self.__videoList)
        # self.trainmodelthread.signal.connect(self.trainModel)
        self.detectvideothread.start()

    def receiveLogForDetectVideo(self, msg):
        """
        接受log信息
        :param msg: log信息
        :return:
        """
        self.textedit.append(msg)
        self.detectvideothread.stop()
        self.action_openvideos_choosevideos.setEnabled(True)
        self.action_openvideos_choosefolder.setEnabled(True)
        self.action_openvideos_clear.setEnabled(True)
        self.action_keyframe.setEnabled(True)
        self.action_feature.setEnabled(True)
        self.action_trainmodel.setEnabled(True)
        self.action_detect.setEnabled(True)

    def lookuphelpfile(self):
        """
        使用系统的浏览器打开帮助文档
        :return:
        """
        os.system("firefox /home/sunbite/PycharmProjects/myFirstPoint/help.html")

    def lookupaboutfile(self):
        """
        使用系统的浏览器打开关于文档
        :return:
        """
        os.system("firefox /home/sunbite/PycharmProjects/myFirstPoint/about.html")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MFSSEL_UI()
    w.show()
    sys.exit(app.exec_())
