# -*- coding: utf-8 -*-
from DataPreparation import DataPreparation
import MFSSEL_Utilities as Utilities
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np
from sklearn.externals import joblib
import os


def KFold_MFSSEL(featureNameDirPath):
    # 初始化dataPreparation对象
    dataPreparation = DataPreparation()
    train_test_tuple_list = dataPreparation.getLabelAndNameTupleList_KFold(featureNameDirPath)
    # 每个类别的准确率
    hog_class_1_accuracy_list = []
    hog_class_2_accuracy_list = []
    hog_class_3_accuracy_list = []
    hog_class_4_accuracy_list = []
    hog_class_5_accuracy_list = []
    hog_class_6_accuracy_list = []
    hog_class_7_accuracy_list = []
    hog_class_8_accuracy_list = []
    hog_class_9_accuracy_list = []
    hog_class_10_accuracy_list = []
    hog_class_11_accuracy_list = []

    # 每个类别的准确率
    hsv_class_1_accuracy_list = []
    hsv_class_2_accuracy_list = []
    hsv_class_3_accuracy_list = []
    hsv_class_4_accuracy_list = []
    hsv_class_5_accuracy_list = []
    hsv_class_6_accuracy_list = []
    hsv_class_7_accuracy_list = []
    hsv_class_8_accuracy_list = []
    hsv_class_9_accuracy_list = []
    hsv_class_10_accuracy_list = []
    hsv_class_11_accuracy_list = []
    # 每个类别的准确率
    lbp_class_1_accuracy_list = []
    lbp_class_2_accuracy_list = []
    lbp_class_3_accuracy_list = []
    lbp_class_4_accuracy_list = []
    lbp_class_5_accuracy_list = []
    lbp_class_6_accuracy_list = []
    lbp_class_7_accuracy_list = []
    lbp_class_8_accuracy_list = []
    lbp_class_9_accuracy_list = []
    lbp_class_10_accuracy_list = []
    lbp_class_11_accuracy_list = []
    for train_test_tuple in train_test_tuple_list:
        print("-----------------------------------------------------------------")
        print(train_test_tuple_list.index(train_test_tuple) + 1)
        print("-----------------------------------------------------------------")
        accuracy_max = 0
        _81FeatureDir = "/_81videoFeature/"
        _30FeatureDir = "/_30videoFeature/"
        hogFeatureDir = "/hogvideoFeature/"

        # 有标签数据集大小list
        label_data_num_list = []
        label_name_labeled_train_tuple_list, label_name_unlabeled_train_tuple_list, label_name_test_tuple_list = train_test_tuple

        # 获取hog维有标签训练集
        hog_labeled_train_tuple_list = dataPreparation.loadData(featureNameDirPath, hogFeatureDir,
                                                                label_name_labeled_train_tuple_list)
        hog_labeled_train_Y, hog_labeled_train_X, hog_labeled_train_Name = Utilities.get_Y_X_Name_list_from_tuple(
            hog_labeled_train_tuple_list)
        # 获取hog维无标签训练集
        hog_unlabeled_tuple_list = dataPreparation.loadData(featureNameDirPath, hogFeatureDir,
                                                            label_name_unlabeled_train_tuple_list)
        hog_unlabeled_Y, hog_unlabeled_X, hog_unlabeled_Name = Utilities.get_Y_X_Name_list_from_tuple(
            hog_unlabeled_tuple_list)
        # 获取hog维测试集
        hog_test_tuple_list = dataPreparation.loadData(featureNameDirPath, hogFeatureDir, label_name_test_tuple_list)
        hog_test_Y, hog_test_X, hog_test_Name = Utilities.get_Y_X_Name_list_from_tuple(hog_test_tuple_list)

        # 获取81维有标签训练集
        _81_labeled_train_tuple_list = dataPreparation.loadData(featureNameDirPath, _81FeatureDir,
                                                                label_name_labeled_train_tuple_list)
        _81_labeled_train_Y, _81_labeled_train_X, _81_labeled_train_Name = Utilities.get_Y_X_Name_list_from_tuple(
            _81_labeled_train_tuple_list)
        # 获取81维无标签训练集
        _81_unlabeled_tuple_list = dataPreparation.loadData(featureNameDirPath, _81FeatureDir,
                                                            label_name_unlabeled_train_tuple_list)
        _81_unlabeled_Y, _81_unlabeled_X, _81_unlabeled_Name = Utilities.get_Y_X_Name_list_from_tuple(
            _81_unlabeled_tuple_list)
        # 获取81维测试集
        _81_test_tuple_list = dataPreparation.loadData(featureNameDirPath, _81FeatureDir, label_name_test_tuple_list)
        _81_test_Y, _81_test_X, _81_test_Name = Utilities.get_Y_X_Name_list_from_tuple(_81_test_tuple_list)

        # 获取30维有标签训练集
        _30_labeled_train_tuple_list = dataPreparation.loadData(featureNameDirPath, _30FeatureDir,
                                                                label_name_labeled_train_tuple_list)
        _30_labeled_train_Y, _30_labeled_train_X, _30_labeled_train_Name = Utilities.get_Y_X_Name_list_from_tuple(
            _30_labeled_train_tuple_list)
        # 获取30维无标签训练集
        _30_unlabeled_tuple_list = dataPreparation.loadData(featureNameDirPath, _30FeatureDir,
                                                            label_name_unlabeled_train_tuple_list)
        _30_unlabeled_Y, _30_unlabeled_X, _30_unlabeled_Name = Utilities.get_Y_X_Name_list_from_tuple(
            _30_unlabeled_tuple_list)
        # 获取30维测试集
        _30_test_tuple_list = dataPreparation.loadData(featureNameDirPath, _30FeatureDir, label_name_test_tuple_list)
        _30_test_Y, _30_test_X, _30_test_Name = Utilities.get_Y_X_Name_list_from_tuple(_30_test_tuple_list)

        # hog维svm训练
        hog_svc_1 = SVC(C=4, kernel='rbf', gamma=2, probability=True)  # c:4 gamma=2
        hog_svc_1.fit(hog_labeled_train_X, hog_labeled_train_Y)

        # 81维svm训练
        _81_svc_1 = SVC(C=4, kernel='rbf', gamma=14, probability=True)  # c:2 gamma=6
        _81_svc_1.fit(_81_labeled_train_X, _81_labeled_train_Y)

        # 30维svm训练
        _30_svc_1 = SVC(C=32, kernel='rbf', gamma=8, probability=True)  # c:32 gamma=12
        _30_svc_1.fit(_30_labeled_train_X, _30_labeled_train_Y)

        whole_labeled_train_X = hog_labeled_train_X.copy()
        whole_labeled_train_X = np.concatenate([whole_labeled_train_X, _81_labeled_train_X], axis=1)
        whole_labeled_train_X = np.concatenate([whole_labeled_train_X, _30_labeled_train_X], axis=1)

        # hog维svm训练
        whole_svc_1 = SVC(C=4, kernel='rbf', gamma=2, probability=True)  # c:4 gamma=2
        whole_svc_1.fit(whole_labeled_train_X, hog_labeled_train_Y)

        whole_test_X = hog_test_X.copy()
        whole_test_X = np.concatenate([whole_test_X, _81_test_X], axis=1)
        whole_test_X = np.concatenate([whole_test_X, _30_test_X], axis=1)
        label_data_num_list.append(len(hog_labeled_train_Y))
        # 获得准确率
        hog_accuracy = hog_svc_1.score(hog_test_X, hog_test_Y)
        # 获得准确率
        _81_accuracy = _81_svc_1.score(_81_test_X, _81_test_Y)
        # 获得准确率
        _30_accuracy = _30_svc_1.score(_30_test_X, _30_test_Y)

        # 获得准确率
        whole_accuracy = whole_svc_1.score(whole_test_X, _30_test_Y)

        # hog特征下的无标签数据集的所对应的各个类别的概率
        hog_svc_1_test_probility = hog_svc_1.predict_proba(hog_test_X)
        # hog特征下无标签数据的预测标签
        hog_svc_1_test_predict_Y = hog_svc_1.predict(hog_test_X)

        # 81维特征下的无标签数据集的所对应的各个类别的概率
        _81_svc_1_test_probility = _81_svc_1.predict_proba(_81_test_X)
        # 81维特征下无标签数据的预测标签
        _81_svc_1_test_predict_Y = _81_svc_1.predict(_81_test_X)

        # 30维特征下的无标签数据集的所对应的各个类别的概率
        _30_svc_1_test_probility = _30_svc_1.predict_proba(_30_test_X)
        # 30维特征下无标签数据的预测标签
        _30_svc_1_test_predict_Y = _30_svc_1.predict(_30_test_X)

        each_class_hog_accuracy_list = Utilities.get_each_class_accuracy(hog_svc_1_test_predict_Y, hog_test_Y)
        each_class_hsv_accuracy_list = Utilities.get_each_class_accuracy(_81_svc_1_test_predict_Y, hog_test_Y)
        each_class_lbp_accuracy_list = Utilities.get_each_class_accuracy(_30_svc_1_test_predict_Y, hog_test_Y)

        hog_class_1_accuracy_list.append(each_class_hog_accuracy_list[0])
        hog_class_2_accuracy_list.append(each_class_hog_accuracy_list[1])
        hog_class_3_accuracy_list.append(each_class_hog_accuracy_list[2])
        hog_class_4_accuracy_list.append(each_class_hog_accuracy_list[3])
        hog_class_5_accuracy_list.append(each_class_hog_accuracy_list[4])
        hog_class_6_accuracy_list.append(each_class_hog_accuracy_list[5])
        hog_class_7_accuracy_list.append(each_class_hog_accuracy_list[6])
        hog_class_8_accuracy_list.append(each_class_hog_accuracy_list[7])
        hog_class_9_accuracy_list.append(each_class_hog_accuracy_list[8])
        hog_class_10_accuracy_list.append(each_class_hog_accuracy_list[9])
        hog_class_11_accuracy_list.append(each_class_hog_accuracy_list[10])

        hsv_class_1_accuracy_list.append(each_class_hsv_accuracy_list[0])
        hsv_class_2_accuracy_list.append(each_class_hsv_accuracy_list[1])
        hsv_class_3_accuracy_list.append(each_class_hsv_accuracy_list[2])
        hsv_class_4_accuracy_list.append(each_class_hsv_accuracy_list[3])
        hsv_class_5_accuracy_list.append(each_class_hsv_accuracy_list[4])
        hsv_class_6_accuracy_list.append(each_class_hsv_accuracy_list[5])
        hsv_class_7_accuracy_list.append(each_class_hsv_accuracy_list[6])
        hsv_class_8_accuracy_list.append(each_class_hsv_accuracy_list[7])
        hsv_class_9_accuracy_list.append(each_class_hsv_accuracy_list[8])
        hsv_class_10_accuracy_list.append(each_class_hsv_accuracy_list[9])
        hsv_class_11_accuracy_list.append(each_class_hsv_accuracy_list[10])

        lbp_class_1_accuracy_list.append(each_class_lbp_accuracy_list[0])
        lbp_class_2_accuracy_list.append(each_class_lbp_accuracy_list[1])
        lbp_class_3_accuracy_list.append(each_class_lbp_accuracy_list[2])
        lbp_class_4_accuracy_list.append(each_class_lbp_accuracy_list[3])
        lbp_class_5_accuracy_list.append(each_class_lbp_accuracy_list[4])
        lbp_class_6_accuracy_list.append(each_class_lbp_accuracy_list[5])
        lbp_class_7_accuracy_list.append(each_class_lbp_accuracy_list[6])
        lbp_class_8_accuracy_list.append(each_class_lbp_accuracy_list[7])
        lbp_class_9_accuracy_list.append(each_class_lbp_accuracy_list[8])
        lbp_class_10_accuracy_list.append(each_class_lbp_accuracy_list[9])
        lbp_class_11_accuracy_list.append(each_class_lbp_accuracy_list[10])

        # max_each_class_hog_accuracy_list = max(each_class_hog_accuracy_list)
        # min_each_class_hog_accuracy_list = min(each_class_hog_accuracy_list)
        # max_each_class_hsv_accuracy_list = max(each_class_hsv_accuracy_list)
        # min_each_class_hsv_accuracy_list = min(each_class_hsv_accuracy_list)
        # max_each_class_lbp_accuracy_list = max(each_class_lbp_accuracy_list)
        # min_each_class_lbp_accuracy_list = min(each_class_lbp_accuracy_list)

        print("SVM-hog准确率：")
        print(hog_accuracy * 100)
        print("SVM-hsv准确率：")
        print(_81_accuracy * 100)
        print("SVM-lbp准确率：")
        print(_30_accuracy * 100)
        print("SVM-whole准确率：")
        print(whole_accuracy * 100)
        print("有标签数据集大小：")
        print(label_data_num_list)
    # print("SVM-hog_class_1_accuracy_list:")
    # print(sum(hog_class_1_accuracy_list)/len(hog_class_1_accuracy_list))
    # print("SVM-hog_class_2_accuracy_list:")
    # print(sum(hog_class_2_accuracy_list) / len(hog_class_2_accuracy_list))
    # print("SVM-hog_class_3_accuracy_list:")
    # print(sum(hog_class_3_accuracy_list) / len(hog_class_3_accuracy_list))
    # print("SVM-hog_class_4_accuracy_list:")
    # print(sum(hog_class_4_accuracy_list) / len(hog_class_4_accuracy_list))
    # print("SVM-hog_class_5_accuracy_list:")
    # print(sum(hog_class_5_accuracy_list) / len(hog_class_5_accuracy_list))
    # print("SVM-hog_class_6_accuracy_list:")
    # print(sum(hog_class_6_accuracy_list) / len(hog_class_6_accuracy_list))
    # print("SVM-hog_class_7_accuracy_list:")
    # print(sum(hog_class_7_accuracy_list) / len(hog_class_7_accuracy_list))
    # print("SVM-hog_class_8_accuracy_list:")
    # print(sum(hog_class_8_accuracy_list) / len(hog_class_8_accuracy_list))
    # print("SVM-hog_class_9_accuracy_list:")
    # print(sum(hog_class_9_accuracy_list) / len(hog_class_9_accuracy_list))
    # print("SVM-hog_class_10_accuracy_list:")
    # print(sum(hog_class_10_accuracy_list) / len(hog_class_10_accuracy_list))
    # print("SVM-hog_class_11_accuracy_list:")
    # print(sum(hog_class_11_accuracy_list) / len(hog_class_11_accuracy_list))
    #
    # print("SVM-hsv_class_1_accuracy_list:")
    # print(sum(hsv_class_1_accuracy_list) / len(hsv_class_1_accuracy_list))
    # print("SVM-hsv_class_2_accuracy_list:")
    # print(sum(hsv_class_2_accuracy_list) / len(hsv_class_2_accuracy_list))
    # print("SVM-hsv_class_3_accuracy_list:")
    # print(sum(hsv_class_3_accuracy_list) / len(hsv_class_3_accuracy_list))
    # print("SVM-hsv_class_4_accuracy_list:")
    # print(sum(hsv_class_4_accuracy_list) / len(hsv_class_4_accuracy_list))
    # print("SVM-hsv_class_5_accuracy_list:")
    # print(sum(hsv_class_5_accuracy_list) / len(hsv_class_5_accuracy_list))
    # print("SVM-hsv_class_6_accuracy_list:")
    # print(sum(hsv_class_6_accuracy_list) / len(hsv_class_6_accuracy_list))
    # print("SVM-hsv_class_7_accuracy_list:")
    # print(sum(hsv_class_7_accuracy_list) / len(hsv_class_7_accuracy_list))
    # print("SVM-hsv_class_8_accuracy_list:")
    # print(sum(hsv_class_8_accuracy_list) / len(hsv_class_8_accuracy_list))
    # print("SVM-hsv_class_9_accuracy_list:")
    # print(sum(hsv_class_9_accuracy_list) / len(hsv_class_9_accuracy_list))
    # print("SVM-hsv_class_10_accuracy_list:")
    # print(sum(hsv_class_10_accuracy_list) / len(hsv_class_10_accuracy_list))
    # print("SVM-hsv_class_11_accuracy_list:")
    # print(sum(hsv_class_11_accuracy_list) / len(hsv_class_11_accuracy_list))
    #
    # print("SVM-lbp_class_1_accuracy_list:")
    # print(sum(lbp_class_1_accuracy_list) / len(lbp_class_1_accuracy_list))
    # print("SVM-lbp_class_2_accuracy_list:")
    # print(sum(lbp_class_2_accuracy_list) / len(lbp_class_2_accuracy_list))
    # print("SVM-lbp_class_3_accuracy_list:")
    # print(sum(lbp_class_3_accuracy_list) / len(lbp_class_3_accuracy_list))
    # print("SVM-lbp_class_4_accuracy_list:")
    # print(sum(lbp_class_4_accuracy_list) / len(lbp_class_4_accuracy_list))
    # print("SVM-lbp_class_5_accuracy_list:")
    # print(sum(lbp_class_5_accuracy_list) / len(lbp_class_5_accuracy_list))
    # print("SVM-lbp_class_6_accuracy_list:")
    # print(sum(lbp_class_6_accuracy_list) / len(lbp_class_6_accuracy_list))
    # print("SVM-lbp_class_7_accuracy_list:")
    # print(sum(lbp_class_7_accuracy_list) / len(lbp_class_7_accuracy_list))
    # print("SVM-lbp_class_8_accuracy_list:")
    # print(sum(lbp_class_8_accuracy_list) / len(lbp_class_8_accuracy_list))
    # print("SVM-lbp_class_9_accuracy_list:")
    # print(sum(lbp_class_9_accuracy_list) / len(lbp_class_9_accuracy_list))
    # print("SVM-lbp_class_10_accuracy_list:")
    # print(sum(lbp_class_10_accuracy_list) / len(lbp_class_10_accuracy_list))
    # print("SVM-lbp_class_11_accuracy_list:")
    # print(sum(lbp_class_11_accuracy_list) / len(lbp_class_11_accuracy_list))


if __name__ == '__main__':
    KFold_MFSSEL("/home/sunbite/MFSSEL/features_new_2/")
