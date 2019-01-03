# -*- coding: utf-8 -*-
import cv2
import os
import math
import GetFeatures
from sklearn.externals import joblib
import MFSSEL_Utilities as Utilities


class VideoDetector:
    def __init__(self, filepath, modelpath, resizeheight=240, resizewidth=320):
        # 保存关键帧列表
        self.__frames = []
        self.__81features = []
        self.__30features = []
        self.__hogfeatures = []
        self.__filepath = filepath
        self.__modelpath = modelpath
        self.__resizeheight = resizeheight
        self.__resizewidth = resizewidth
        self.__classmap = {"shooting": 1, "biking": 2, "diving": 3, "golf": 4, "riding": 5, "juggle": 6, "swing": 7,
                           "tennis": 8, "jumping": 9, "spiking": 10, "walk": 11}

    def extractkeyframe(self):
        """
        提取关键帧
        """
        if os.path.exists(self.__filepath):
            videocapture = cv2.VideoCapture(self.__filepath)
            if videocapture.isOpened():
                # 所有帧
                wholeframenum = int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT))
                if wholeframenum < 2:
                    print('the image you inputted has not enougth frames!')
                # 中间帧
                middleframenum = math.ceil(wholeframenum / 2)
                # 获取第一帧
                success, frame = videocapture.read()
                frame = cv2.resize(frame, (self.__resizewidth, self.__resizeheight))
                # 加入第一帧filepathandname
                self.__frames = [frame]
                count = 0
                while success:
                    count += 1
                    # 获取下一帧
                    success, frame = videocapture.read()
                    if success:
                        frame = cv2.resize(frame, (self.__resizewidth, self.__resizeheight))
                        # 加入中间帧
                        if count == middleframenum:
                            self.__frames.append(frame)
                        # 加入最后一帧
                        elif count == wholeframenum - 1:
                            self.__frames.append(frame)
                self.__frames.insert(0, self.__filepath)
        else:
            print(" you inputted file is not existed!")

    def getFeatures(self):
        """
        获得三帧的对应的特征
        :return:
        """
        self.extractkeyframe()
        if len(self.__frames) == 4:
            _81features = []
            _30features = []
            hogfeatures = []
            for i in range(0, len(self.__frames) - 1):
                # 获取关键帧的特征
                myFeature = GetFeatures.get_features(self.__frames[i + 1])
                # 获取不同组合的特征
                _81feature = GetFeatures._81feature(myFeature)
                _30feature = GetFeatures._30feature(myFeature)
                hogfeature = GetFeatures.hog_feature(self.__frames[i + 1])
                _81features.extend(_81feature)
                _30features.extend(_30feature)
                hogfeatures.extend(hogfeature)
            self.__81features = self.getFeaturesList(_81features)
            self.__30features = self.getFeaturesList(_30features)
            self.__hogfeatures = self.getFeaturesList(hogfeatures)
            # classname = os.path.basename(os.path.dirname(self.__frames[0]))
            classname = os.path.basename(self.__frames[0]).split("_")[1]
            classnum = self.__classmap[classname]
            # 把真正的标签放在特征list的第一个位置
            self.__81features.insert(0, float(classnum))
            self.__30features.insert(0, float(classnum))
            self.__hogfeatures.insert(0, float(classnum))
        else:
            print("your keyframes are not enough!!")
        return self.__81features, self.__30features, self.__hogfeatures

    def getLabel(self):
        """
        获取label
        :return:
        """
        self.getFeatures()
        if os.path.exists(self.__modelpath):
            # 加载model文件
            hogModel = joblib.load(self.__modelpath + "/hog.model")
            _81Model = joblib.load(self.__modelpath + "/81.model")
            _30Model = joblib.load(self.__modelpath + "/30.model")
            test_y = [self.__hogfeatures[0]]
            test_x_hog = [self.__hogfeatures[1:]]
            test_x_81 = [self.__81features[1:]]
            test_x_30 = [self.__30features[1:]]
            # hog特征下的无标签数据集的所对应的各个类别的概率
            hog_svc_1_test_probility = hogModel.predict_proba(test_x_hog)
            # hog特征下无标签数据的预测标签
            hog_svc_1_test_predict_Y = hogModel.predict(test_x_hog)

            # 81维特征下的无标签数据集的所对应的各个类别的概率
            _81_svc_1_test_probility = _81Model.predict_proba(test_x_81)
            # 81维特征下无标签数据的预测标签
            _81_svc_1_test_predict_Y = _81Model.predict(test_x_81)

            # 30维特征下的无标签数据集的所对应的各个类别的概率
            _30_svc_1_test_probility = _30Model.predict_proba(test_x_30)
            # 30维特征下无标签数据的预测标签
            _30_svc_1_test_predict_Y = _30Model.predict(test_x_30)

            mfssel_predict_Y_list_ = Utilities.predict_Y(hog_svc_1_test_probility, _81_svc_1_test_probility,
                                                         _30_svc_1_test_probility, hog_svc_1_test_predict_Y,
                                                         _81_svc_1_test_predict_Y, _30_svc_1_test_predict_Y)
            return mfssel_predict_Y_list_[0], test_y[0]

    def getFeaturesList(self, my_feature):
        """
        获取特征list
        :return:
        """
        my_feature_temp = []
        for i in range(0, len(my_feature)):
                my_feature_temp.append(float(my_feature[i]))
        return my_feature_temp
