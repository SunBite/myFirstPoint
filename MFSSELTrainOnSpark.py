# -*- coding: utf-8 -*-
from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel
import datetime
import os
from ListParam import *
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import MFSSEL_Utilities as Utilities
from sklearn.svm import SVC
from sklearn.externals import joblib


class MFSSELTrainOnSpark:

    def __init__(self, filepath, savepath):
        """
        初始化方法
        :param filepath: hdfs上的要读取的features目录
        :param savepath: 保存的地址
        """
        self.__filepath = filepath
        self.__savepath = savepath
        self.__classmap = {"shooting": float(1), "biking": float(2), "diving": float(3),
                           "golf": float(4), "riding": float(5), "juggle": float(6),
                           "swing": float(7), "tennis": float(8), "jumping": float(9),
                           "spiking": float(10), "walk": float(11)}

    def MFSSELTrainOnSpark(self):
        """
        训练模型，预测结果
        """
        global TOTALFEATURESANDLABEL_hsv
        global TOTALFEATURESANDLABEL_lbp
        global TOTALFEATURESANDLABEL_hog
        _81FeaturesPath = self.__filepath + "hsvVideoFeature/"
        _30FeaturesPath = self.__filepath + "lbpVideoFeature/"
        hogFeaturesPath = self.__filepath + "hogVideoFeature/"
        sc = SparkContext(appName="MFSSELTrainOnSpark")
        TOTALFEATURESANDLABEL_hsv = sc.accumulator([], ListParamForFeatureAndLabel())
        TOTALFEATURESANDLABEL_lbp = sc.accumulator([], ListParamForFeatureAndLabel())
        TOTALFEATURESANDLABEL_hog = sc.accumulator([], ListParamForFeatureAndLabel())

        _81Features = sc.textFile(_81FeaturesPath)
        _30Features = sc.textFile(_30FeaturesPath)
        hogFeatures = sc.textFile(hogFeaturesPath)

        def makeFeaturesName(line):
            """
            选出特征名字
            :param line:关键帧的特征
            """
            features = line[1]
            featureNameList = os.path.basename(line[0]).split("_v")[1].split("_")[1:-1]
            featureName = "_".join(featureNameList)
            return (featureName, features)

        def getTotalFeatures_hsv(line):
            """
            累加hsv特征到累加器
            """
            global TOTALFEATURESANDLABEL_hsv
            TOTALFEATURESANDLABEL_hsv += [(line[0], line[1])]

        def getTotalFeatures_lbp(line):
            """
            累加lbp特征到累加器
            """
            global TOTALFEATURESANDLABEL_lbp
            TOTALFEATURESANDLABEL_lbp += [(line[0], line[1])]

        def getTotalFeatures_hog(line):
            """
            累加hog特征到累加器
            """
            global TOTALFEATURESANDLABEL_hog
            TOTALFEATURESANDLABEL_hog += [(line[0], line[1])]

        _81Features.map(lambda x: x.split(" ")).map(lambda x: (x[0], x[1:])).map(makeFeaturesName).map(
            getTotalFeatures_hsv).persist(storageLevel=StorageLevel.MEMORY_ONLY).count()
        _30Features.map(lambda x: x.split(" ")).map(lambda x: (x[0], x[1:])).map(makeFeaturesName).map(
            getTotalFeatures_lbp).persist(storageLevel=StorageLevel.MEMORY_ONLY).count()
        hogFeatures.map(lambda x: x.split(" ")).map(lambda x: (x[0], x[1:])).map(makeFeaturesName).map(
            getTotalFeatures_hog).persist(storageLevel=StorageLevel.MEMORY_ONLY).count()
        totalfeaturesandlabel_hsv = TOTALFEATURESANDLABEL_hsv.value
        totalfeaturesandlabel_lbp = TOTALFEATURESANDLABEL_lbp.value
        totalfeaturesandlabel_hog = TOTALFEATURESANDLABEL_hog.value

        def getfeaturelistandlabellist(totalfeaturesandlabel):
            """
            把累加器中的label和特征的元组提出来，形成标签list和featrueslist
            :param totalfeaturesandlabel:label和特征的元组
            :return:（标签list，featrueslist）
            """
            TOTALFEATURES = []
            TOTALNAME = []

            for i in range(0, len(totalfeaturesandlabel)):
                TOTALNAME.append(totalfeaturesandlabel[i][0])
                TOTALFEATURES.append(totalfeaturesandlabel[i][1])
            return (TOTALNAME, TOTALFEATURES)

        def makeOrder(totalfeaturesandlabel_hsv, totalfeaturesandlabel_lbp, totalfeaturesandlabel_hog):
            """
            根据hsv特征向量list的顺序，调整lbp和hog特征向量list的顺序
            """

            totalname_hsv, totalfeatures_hsv = getfeaturelistandlabellist(totalfeaturesandlabel_hsv)
            totalname_lbp, totalfeatures_lbp = getfeaturelistandlabellist(totalfeaturesandlabel_lbp)
            totalname_hog, totalfeatures_hog = getfeaturelistandlabellist(totalfeaturesandlabel_hog)
            index_name_lbp = []
            index_name_hog = []
            for name in totalname_hsv:
                index_name_lbp.append(totalname_lbp.index(name))
                index_name_hog.append(totalname_hog.index(name))
            totalname_hsv = np.array(totalname_hsv)
            totalfeatures_hsv = np.array(totalfeatures_hsv)
            totalfeatures_lbp = np.array(totalfeatures_lbp)
            totalfeatures_hog = np.array(totalfeatures_hog)
            totalfeatures_lbp_new = totalfeatures_lbp[index_name_lbp]
            totalfeatures_hog_new = totalfeatures_hog[index_name_hog]

            totallabel_new = []
            for name in totalname_hsv:
                classname = name.split("_")[0]
                classnum = self.__classmap[classname]
                totallabel_new.append(classnum)
            totallabel_new = np.array(totallabel_new)
            return totallabel_new, totalfeatures_hsv, totallabel_new, totalfeatures_lbp_new, totallabel_new, totalfeatures_hog_new

        totallabel_hsv, totalfeatures_hsv, totallabel_lbp, totalfeatures_lbp, totallabel_hog, totalfeatures_hog = makeOrder(
            totalfeaturesandlabel_hsv, totalfeaturesandlabel_lbp, totalfeaturesandlabel_hog)

        # print(totallabel_hog)
        # print(totallabel_lbp)
        # # 每个类别的初始准确率
        # first_class_1_accuracy_list = []
        # first_class_2_accuracy_list = []
        # first_class_3_accuracy_list = []
        # first_class_4_accuracy_list = []
        # first_class_5_accuracy_list = []
        # first_class_6_accuracy_list = []
        # first_class_7_accuracy_list = []
        # first_class_8_accuracy_list = []
        # first_class_9_accuracy_list = []
        # first_class_10_accuracy_list = []
        # first_class_11_accuracy_list = []
        # # 每个类别的最终准确率
        # class_1_accuracy_list = []
        # class_2_accuracy_list = []
        # class_3_accuracy_list = []
        # class_4_accuracy_list = []
        # class_5_accuracy_list = []
        # class_6_accuracy_list = []
        # class_7_accuracy_list = []
        # class_8_accuracy_list = []
        # class_9_accuracy_list = []
        # class_10_accuracy_list = []
        # class_11_accuracy_list = []
        # # 整体初始准确率
        # whole_first_accuracy_list = []
        # # 整体准确率
        # whole_accuracy_list = []

        sfolder = StratifiedKFold(n_splits=10, random_state=0, shuffle=False)  # 2 5
        KFold_Num = 0
        for train_index, test_index in sfolder.split(totalfeatures_hsv, totallabel_hsv):
            KFold_Num = KFold_Num + 1
            print("-----------------------------------------------------------------")
            print(KFold_Num)
            print("-----------------------------------------------------------------")
            X_train_hsv, X_test_hsv = totalfeatures_hsv[train_index], totalfeatures_hsv[test_index]
            Y_train_hsv, Y_test_hsv = totallabel_hsv[train_index], totallabel_hsv[test_index]
            X_train_lbp, X_test_lbp = totalfeatures_lbp[train_index], totalfeatures_lbp[test_index]
            Y_train_lbp, Y_test_lbp = totallabel_lbp[train_index], totallabel_lbp[test_index]
            X_train_hog, X_test_hog = totalfeatures_hog[train_index], totalfeatures_hog[test_index]
            Y_train_hog, Y_test_hog = totallabel_hog[train_index], totallabel_hog[test_index]


            unlabeled_x_hsv, labeled_x_hsv, unlabeled_y_hsv, labeled_y_hsv = train_test_split(X_train_hsv, Y_train_hsv,
                                                                                              test_size=0.54,
                                                                                              stratify=Y_train_hsv,
                                                                                              random_state=2)
            unlabeled_x_lbp, labeled_x_lbp, unlabeled_y_lbp, labeled_y_lbp = train_test_split(X_train_lbp, Y_train_lbp,
                                                                                              test_size=0.54,
                                                                                              stratify=Y_train_lbp,
                                                                                              random_state=2)
            unlabeled_x_hog, labeled_x_hog, unlabeled_y_hog, labeled_y_hog = train_test_split(X_train_hog, Y_train_hog,
                                                                                              test_size=0.54,
                                                                                              stratify=Y_train_hog,
                                                                                              random_state=2)
            model_max_hog = None
            model_max_81 = None
            model_max_30 = None
            accuracy_max = 0
            # 准确率
            accuracy_list = []
            # 整体类别
            whole_class = 11
            # 选取置信度前topk个样本
            topk = 5
            # 迭代次数
            loop_num = 100
            num = 0.7
            para = 0.7

            for h in range(1, loop_num + 1):
                # pass
                # accuracy_svm_list = []
                # C = [1, 2, 4, 8, 16, 32, 64, 128]
                # gamma = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                # for i in C:
                #     for j in gamma:
                #         # 30维svm训练
                #         orbsvc_1 = SVC(C=i, kernel='rbf', gamma=j, probability=True)  # gamma=7
                #         orbsvc_1.fit(labeled_x_hog, labeled_y_hog)
                #         # 获得准确率
                #         orbaccuracy = orbsvc_1.score(X_test_hog, Y_test_hog)
                #         a = "c:" + str(i) + "   gamma:" + str(j) + "   accuracy" + str(orbaccuracy)
                #         print(a)
                #         accuracy_svm_list.append(a)
                # hog维svm训练
                hog_svc_1 = SVC(C=4, kernel='rbf', gamma=1, probability=True)  # c:4 gamma=2
                hog_svc_1.fit(labeled_x_hog, labeled_y_hog)

                # 81维svm训练
                _81_svc_1 = SVC(C=2, kernel='rbf', gamma=13, probability=True)  # c:2 gamma=13 c:16 gamma=3
                _81_svc_1.fit(labeled_x_hsv, labeled_y_hsv)

                # 30维svm训练
                _30_svc_1 = SVC(C=32, kernel='rbf', gamma=1, probability=True)  # c:32 gamma=12 c:32 gamma=1
                _30_svc_1.fit(labeled_x_lbp, labeled_y_lbp)

                # 获得准确率
                hog_accuracy = hog_svc_1.score(X_test_hog, Y_test_hog)
                # 获得准确率
                _81_accuracy = _81_svc_1.score(X_test_hsv, Y_test_hsv)
                # 获得准确率
                _30_accuracy = _30_svc_1.score(X_test_lbp, Y_test_lbp)

                print("hog_accuracy:")
                print(hog_accuracy)
                print("_81_accuracy:")
                print(_81_accuracy)
                print("_30_accuracy:")
                print(_30_accuracy)

                if h == 1:
                    if hog_svc_1 is not None:
                        print("正在保存first_hog.model...")
                        joblib.dump(hog_svc_1, self.__savepath + "first_hog.model")
                        print("保存first_hog.model完毕。")
                    if _81_svc_1 is not None:
                        print("正在保存first_81.model...")
                        joblib.dump(_81_svc_1, self.__savepath + "first_81.model")
                        print("保存first_81.model完毕。")
                    if _30_svc_1 is not None:
                        print("正在保存first_30.model...")
                        joblib.dump(_30_svc_1, self.__savepath + "first_30.model")
                        print("保存first_30.model完毕。")
                accuracy_max_from_single = max(hog_accuracy, _81_accuracy, _30_accuracy)
                accuracy_list.append(accuracy_max_from_single * 100)
                print(accuracy_list)
                if accuracy_max_from_single > accuracy_max:
                    accuracy_max = accuracy_max_from_single
                    model_max_hog = hog_svc_1
                    model_max_81 = _81_svc_1
                    model_max_30 = _30_svc_1

                if len(unlabeled_x_lbp) == 0 or len(unlabeled_x_hsv) == 0 or len(unlabeled_x_hog) == 0:
                    break
                if h == loop_num:
                    break

                # hog特征下的测试数据集的所对应的各个类别的概率
                hog_svc_1_probility = hog_svc_1.predict_proba(unlabeled_x_hog)
                # hog特征下测试数据集的预测标签
                hog_svc_1_predict_Y = hog_svc_1.predict(unlabeled_x_hog)

                # 81维特征下的测试数据集的所对应的各个类别的概率
                _81_svc_1_probility = _81_svc_1.predict_proba(unlabeled_x_hsv)
                # 81维特征下测试数据集的预测标签
                _81_svc_1_predict_Y = _81_svc_1.predict(unlabeled_x_hsv)

                # 30维特征下的测试数据集的所对应的各个类别的概率
                _30_svc_1_probility = _30_svc_1.predict_proba(unlabeled_x_lbp)
                # 30维特征下测试数据集的预测标签
                _30_svc_1_predict_Y = _30_svc_1.predict(unlabeled_x_lbp)

                probility_list_1 = [hog_svc_1_probility, _81_svc_1_probility, _30_svc_1_probility]
                unlabeled_Y_list_1 = [unlabeled_y_hog, unlabeled_y_hsv, unlabeled_y_lbp]
                predict_Y_list_1 = [hog_svc_1_predict_Y, _81_svc_1_predict_Y, _30_svc_1_predict_Y]

                selected_ind_list, selected_pesudo_label_list = Utilities.get_pesudo_label(probility_list_1,
                                                                                           predict_Y_list_1,
                                                                                           unlabeled_Y_list_1,
                                                                                           whole_class,
                                                                                           topk, num, para)
                labeled_x_hog = list(labeled_x_hog)
                labeled_y_hog = list(labeled_y_hog)
                labeled_x_hsv = list(labeled_x_hsv)
                labeled_y_hsv = list(labeled_y_hsv)
                labeled_x_lbp = list(labeled_x_lbp)
                labeled_y_lbp = list(labeled_y_lbp)

                for i in range(len(selected_ind_list)):
                    labeled_x_hog.append(unlabeled_x_hog[selected_ind_list[i]])
                    labeled_y_hog.append(selected_pesudo_label_list[i])
                    labeled_x_hsv.append(unlabeled_x_hsv[selected_ind_list[i]])
                    labeled_y_hsv.append(selected_pesudo_label_list[i])
                    labeled_x_lbp.append(unlabeled_x_lbp[selected_ind_list[i]])
                    labeled_y_lbp.append(selected_pesudo_label_list[i])

                unlabeled_x_hog = [i for j, i in enumerate(unlabeled_x_hog) if j not in selected_ind_list]
                unlabeled_y_hog = [i for j, i in enumerate(unlabeled_y_hog) if j not in selected_ind_list]
                unlabeled_x_hsv = [i for j, i in enumerate(unlabeled_x_hsv) if j not in selected_ind_list]
                unlabeled_y_hsv = [i for j, i in enumerate(unlabeled_y_hsv) if j not in selected_ind_list]
                unlabeled_x_lbp = [i for j, i in enumerate(unlabeled_x_lbp) if j not in selected_ind_list]
                unlabeled_y_lbp = [i for j, i in enumerate(unlabeled_y_lbp) if j not in selected_ind_list]

            if model_max_hog is not None:
                print("正在保存hog.model...")
                joblib.dump(model_max_hog, self.__savepath + "hog.model")
                print("保存hog.model完毕。")
            if model_max_81 is not None:
                print("正在保存81.model...")
                joblib.dump(model_max_81, self.__savepath + "81.model")
                print("保存81.model完毕。")
            if model_max_30 is not None:
                print("正在保存30.model...")
                joblib.dump(model_max_30, self.__savepath + "30.model")
                print("保存30.model完毕。")


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    # ckstapos.CoKNNSVMTrainAndPredictOnSpark("hdfs://sunbite-computer:9000/filepath/filepath320240-366.txt",
    MFSSELTrainOnSpark("file:///home/sunbite/MFSSEL/features_on_spark_backup/",
                       '/home/sunbite/MFSSEL/model_on_spark/').MFSSELTrainOnSpark()
    endtime = datetime.datetime.now()
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
    print(
        '----------MFSSELTrainOnSpark Running time: %s Seconds-----------' % (endtime - starttime).seconds)
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
