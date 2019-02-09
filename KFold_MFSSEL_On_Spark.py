# -*- coding: utf-8 -*-
from DataPreparation import DataPreparation
import MFSSEL_Utilities as Utilities
from sklearn.svm import SVC
from sklearn.externals import joblib
import os
import datetime


def KFold_MFSSEL_On_Spark(X_train_hsv, Y_train_hsv, X_test_hsv, Y_test_hsv, X_train_lbp, Y_train_lbp, X_test_lbp,
                          Y_test_lbp, X_train_hog, Y_train_hog, X_test_hog, Y_test_hog, savePath=None,
                          trainAndTestFlag="train"):
    # 每个类别的初始准确率
    first_class_1_accuracy_list = []
    first_class_2_accuracy_list = []
    first_class_3_accuracy_list = []
    first_class_4_accuracy_list = []
    first_class_5_accuracy_list = []
    first_class_6_accuracy_list = []
    first_class_7_accuracy_list = []
    first_class_8_accuracy_list = []
    first_class_9_accuracy_list = []
    first_class_10_accuracy_list = []
    first_class_11_accuracy_list = []
    # 每个类别的最终准确率
    class_1_accuracy_list = []
    class_2_accuracy_list = []
    class_3_accuracy_list = []
    class_4_accuracy_list = []
    class_5_accuracy_list = []
    class_6_accuracy_list = []
    class_7_accuracy_list = []
    class_8_accuracy_list = []
    class_9_accuracy_list = []
    class_10_accuracy_list = []
    class_11_accuracy_list = []
    # 整体初始准确率
    whole_first_accuracy_list = []
    # 整体准确率
    whole_accuracy_list = []
    for train_test_tuple in train_test_tuple_list:
        print("-----------------------------------------------------------------")
        print(train_test_tuple_list.index(train_test_tuple) + 1)
        print("-----------------------------------------------------------------")
        model_max_hog = None
        model_max_81 = None
        model_max_30 = None
        accuracy_max = 0
        _81FeatureDir = "/_81videoFeature/"
        _30FeatureDir = "/_30videoFeature/"
        hogFeatureDir = "/hogvideoFeature/"

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

        for h in range(1, loop_num + 1):
            label_data_num_list.append(len(hog_labeled_train_Y))
            # hog维svm训练
            hog_svc_1 = SVC(C=4, kernel='rbf', gamma=2, probability=True)  # c:4 gamma=2
            hog_svc_1.fit(hog_labeled_train_X, hog_labeled_train_Y)

            # 81维svm训练
            _81_svc_1 = SVC(C=4, kernel='rbf', gamma=14, probability=True)  # c:2 gamma=6
            _81_svc_1.fit(_81_labeled_train_X, _81_labeled_train_Y)

            # 30维svm训练
            _30_svc_1 = SVC(C=32, kernel='rbf', gamma=8, probability=True)  # c:32 gamma=12
            _30_svc_1.fit(_30_labeled_train_X, _30_labeled_train_Y)

            # 获得准确率
            hog_accuracy = hog_svc_1.score(hog_test_X, hog_test_Y)
            # 获得准确率
            _81_accuracy = _81_svc_1.score(_81_test_X, _81_test_Y)
            # 获得准确率
            _30_accuracy = _30_svc_1.score(_30_test_X, _30_test_Y)

            if h == 1:
                if hog_svc_1 is not None:
                    print("正在保存first_hog.model...")
                    joblib.dump(hog_svc_1, savePath + "first_hog.model")
                    print("保存first_hog.model完毕。")
                if _81_svc_1 is not None:
                    print("正在保存first_81.model...")
                    joblib.dump(_81_svc_1, savePath + "first_81.model")
                    print("保存first_81.model完毕。")
                if _30_svc_1 is not None:
                    print("正在保存first_30.model...")
                    joblib.dump(_30_svc_1, savePath + "first_30.model")
                    print("保存first_30.model完毕。")

            # hog_accuracy，_81_accuracy，_30_accuracy中最大的accuracy
            accuracy_max_from_single = max(hog_accuracy, _81_accuracy, _30_accuracy)
            accuracy_list.append(accuracy_max_from_single * 100)
            # hog_accuracy_list.append(hog_accuracy * 100)
            # _81_accuracy_list.append(_81_accuracy * 100)
            # _30_accuracy_list.append(_30_accuracy * 100)

            if accuracy_max_from_single > accuracy_max:
                accuracy_max = accuracy_max_from_single
                model_max_hog = hog_svc_1
                model_max_81 = _81_svc_1
                model_max_30 = _30_svc_1

            if len(hog_unlabeled_X) == 0 or len(_81_unlabeled_X) == 0 or len(_30_unlabeled_X) == 0:
                break
            if h == loop_num:
                break
            # print(accuracy_max * 100)
            # hog特征下的测试数据集的所对应的各个类别的概率
            hog_svc_1_probility = hog_svc_1.predict_proba(hog_unlabeled_X)
            # hog特征下测试数据集的预测标签
            hog_svc_1_predict_Y = hog_svc_1.predict(hog_unlabeled_X)

            # 81维特征下的测试数据集的所对应的各个类别的概率
            _81_svc_1_probility = _81_svc_1.predict_proba(_81_unlabeled_X)
            # 81维特征下测试数据集的预测标签
            _81_svc_1_predict_Y = _81_svc_1.predict(_81_unlabeled_X)

            # 30维特征下的测试数据集的所对应的各个类别的概率
            _30_svc_1_probility = _30_svc_1.predict_proba(_30_unlabeled_X)
            # 30维特征下测试数据集的预测标签
            _30_svc_1_predict_Y = _30_svc_1.predict(_30_unlabeled_X)

            probility_list_1 = [hog_svc_1_probility, _81_svc_1_probility, _30_svc_1_probility]
            unlabeled_Y_list_1 = [hog_unlabeled_Y, _81_unlabeled_Y, _30_unlabeled_Y]
            predict_Y_list_1 = [hog_svc_1_predict_Y, _81_svc_1_predict_Y, _30_svc_1_predict_Y]

            # voted_index_predict_Y_list = Utilities.vote(predict_Y_list_1, unlabeled_Y_list_1, whole_class, topk)
            #
            # voted_Y_list, voted_index_list = Utilities.get_voted_confidence(probility_list_1,
            #                                                                 voted_index_predict_Y_list[0],
            #                                                                 voted_index_predict_Y_list[1], whole_class,
            #                                                                 topk)

            selected_ind_list, selected_pesudo_label_list = Utilities.get_pesudo_label(probility_list_1,
                                                                                       predict_Y_list_1,
                                                                                       unlabeled_Y_list_1,
                                                                                       whole_class,
                                                                                       topk, num, para)
            # a = []
            # for i in selected_ind_list:
            #     a.append(_30_unlabeled_Y[i])
            # print(selected_pesudo_label_list)
            # print(a)

            for i in range(len(selected_ind_list)):
                hog_labeled_train_X.append(hog_unlabeled_X[selected_ind_list[i]])
                hog_labeled_train_Y.append(selected_pesudo_label_list[i])
                _81_labeled_train_X.append(_81_unlabeled_X[selected_ind_list[i]])
                _81_labeled_train_Y.append(selected_pesudo_label_list[i])
                _30_labeled_train_X.append(_30_unlabeled_X[selected_ind_list[i]])
                _30_labeled_train_Y.append(selected_pesudo_label_list[i])

            hog_unlabeled_X = [i for j, i in enumerate(hog_unlabeled_X) if j not in selected_ind_list]
            hog_unlabeled_Y = [i for j, i in enumerate(hog_unlabeled_Y) if j not in selected_ind_list]
            _81_unlabeled_X = [i for j, i in enumerate(_81_unlabeled_X) if j not in selected_ind_list]
            _81_unlabeled_Y = [i for j, i in enumerate(_81_unlabeled_Y) if j not in selected_ind_list]
            _30_unlabeled_X = [i for j, i in enumerate(_30_unlabeled_X) if j not in selected_ind_list]
            _30_unlabeled_Y = [i for j, i in enumerate(_30_unlabeled_Y) if j not in selected_ind_list]

        # print(accuracy_max * 100)
        # print("有标签数据集大小：")
        # print(label_data_num_list)
        # print("accuracy_list:")
        # print(accuracy_list)

        if model_max_hog is not None:
            print("正在保存hog.model...")
            joblib.dump(model_max_hog, savePath + "hog.model")
            print("保存hog.model完毕。")
        if model_max_81 is not None:
            print("正在保存81.model...")
            joblib.dump(model_max_81, savePath + "81.model")
            print("保存81.model完毕。")
        if model_max_30 is not None:
            print("正在保存30.model...")
            joblib.dump(model_max_30, savePath + "30.model")
            print("保存30.model完毕。")
        if os.path.exists(savePath):

            # 加载model文件
            first_hogModel = joblib.load(savePath + "/first_hog.model")
            first_81Model = joblib.load(savePath + "/first_81.model")
            first_30Model = joblib.load(savePath + "/first_30.model")

            # hog特征下的无标签数据集的所对应的各个类别的概率
            first_hog_svc_test_probility = first_hogModel.predict_proba(hog_test_X)
            # hog特征下无标签数据的预测标签
            first_hog_svc_test_predict_Y = first_hogModel.predict(hog_test_X)

            # 81维特征下的无标签数据集的所对应的各个类别的概率
            first_81_svc_test_probility = first_81Model.predict_proba(_81_test_X)
            # 81维特征下无标签数据的预测标签
            first_81_svc_test_predict_Y = first_81Model.predict(_81_test_X)

            # 30维特征下的无标签数据集的所对应的各个类别的概率
            first_30_svc_test_probility = first_30Model.predict_proba(_30_test_X)
            # 30维特征下无标签数据的预测标签
            first_30_svc_test_predict_Y = first_30Model.predict(_30_test_X)

            first_probility_list = first_hog_svc_test_probility, first_81_svc_test_probility, first_30_svc_test_probility
            first_predict_Y_list = first_hog_svc_test_predict_Y, first_81_svc_test_predict_Y, first_30_svc_test_predict_Y
            first_mfssel_predict_Y_list = Utilities.predict_Y_test(first_probility_list, first_predict_Y_list,
                                                                   whole_class, para)
            first_each_class_accuracy_list = Utilities.get_each_class_accuracy(first_mfssel_predict_Y_list, hog_test_Y)
            first_class_1_accuracy_list.append(first_each_class_accuracy_list[0])
            first_class_2_accuracy_list.append(first_each_class_accuracy_list[1])
            first_class_3_accuracy_list.append(first_each_class_accuracy_list[2])
            first_class_4_accuracy_list.append(first_each_class_accuracy_list[3])
            first_class_5_accuracy_list.append(first_each_class_accuracy_list[4])
            first_class_6_accuracy_list.append(first_each_class_accuracy_list[5])
            first_class_7_accuracy_list.append(first_each_class_accuracy_list[6])
            first_class_8_accuracy_list.append(first_each_class_accuracy_list[7])
            first_class_9_accuracy_list.append(first_each_class_accuracy_list[8])
            first_class_10_accuracy_list.append(first_each_class_accuracy_list[9])
            first_class_11_accuracy_list.append(first_each_class_accuracy_list[10])
            if len(first_mfssel_predict_Y_list) == len(hog_test_Y):
                rightNum = 0
                for i in range(len(first_mfssel_predict_Y_list)):
                    if first_mfssel_predict_Y_list[i] == hog_test_Y[i]:
                        rightNum = rightNum + 1
                whole_first_accuracy_list.append(rightNum / len(hog_test_Y))
                print("整体初始准确率：")
                print(rightNum / len(hog_test_Y))
            else:
                print("出错！！")
            print("每一类的初始准确率：")
            print(first_each_class_accuracy_list)
            print("初始平均准确率：")
            print(sum(first_each_class_accuracy_list) / len(first_each_class_accuracy_list))
            # 加载model文件
            hogModel = joblib.load(savePath + "/hog.model")
            _81Model = joblib.load(savePath + "/81.model")
            _30Model = joblib.load(savePath + "/30.model")
            # hog特征下的无标签数据集的所对应的各个类别的概率
            hog_svc_test_probility = hogModel.predict_proba(hog_test_X)
            # hog特征下无标签数据的预测标签
            hog_svc_test_predict_Y = hogModel.predict(hog_test_X)

            # 81维特征下的无标签数据集的所对应的各个类别的概率
            _81_svc_test_probility = _81Model.predict_proba(_81_test_X)
            # 81维特征下无标签数据的预测标签
            _81_svc_test_predict_Y = _81Model.predict(_81_test_X)

            # 30维特征下的无标签数据集的所对应的各个类别的概率
            _30_svc_test_probility = _30Model.predict_proba(_30_test_X)
            # 30维特征下无标签数据的预测标签
            _30_svc_test_predict_Y = _30Model.predict(_30_test_X)

            probility_list = hog_svc_test_probility, _81_svc_test_probility, _30_svc_test_probility
            predict_Y_list = hog_svc_test_predict_Y, _81_svc_test_predict_Y, _30_svc_test_predict_Y
            mfssel_predict_Y_list = Utilities.predict_Y_test(probility_list, predict_Y_list, whole_class, para)
            each_class_accuracy_list = Utilities.get_each_class_accuracy(mfssel_predict_Y_list, hog_test_Y)
            class_1_accuracy_list.append(each_class_accuracy_list[0])
            class_2_accuracy_list.append(each_class_accuracy_list[1])
            class_3_accuracy_list.append(each_class_accuracy_list[2])
            class_4_accuracy_list.append(each_class_accuracy_list[3])
            class_5_accuracy_list.append(each_class_accuracy_list[4])
            class_6_accuracy_list.append(each_class_accuracy_list[5])
            class_7_accuracy_list.append(each_class_accuracy_list[6])
            class_8_accuracy_list.append(each_class_accuracy_list[7])
            class_9_accuracy_list.append(each_class_accuracy_list[8])
            class_10_accuracy_list.append(each_class_accuracy_list[9])
            class_11_accuracy_list.append(each_class_accuracy_list[10])
            print("每一类的最终准确率：")
            print(each_class_accuracy_list)
            print("平均准确率：")
            print(sum(each_class_accuracy_list) / len(each_class_accuracy_list))
            if len(mfssel_predict_Y_list) == len(hog_test_Y):
                rightNum = 0
                for i in range(len(mfssel_predict_Y_list)):
                    if mfssel_predict_Y_list[i] == hog_test_Y[i]:
                        rightNum = rightNum + 1
                whole_accuracy_list.append(rightNum / len(hog_test_Y))
                print("整体准确率：")
                print(rightNum / len(hog_test_Y))
            else:
                print("出错！！")
    print("---------------------------------------------------")
    avg_first_class_1_accuracy = sum(first_class_1_accuracy_list) / len(first_class_1_accuracy_list)
    avg_first_class_2_accuracy = sum(first_class_2_accuracy_list) / len(first_class_2_accuracy_list)
    avg_first_class_3_accuracy = sum(first_class_3_accuracy_list) / len(first_class_3_accuracy_list)
    avg_first_class_4_accuracy = sum(first_class_4_accuracy_list) / len(first_class_4_accuracy_list)
    avg_first_class_5_accuracy = sum(first_class_5_accuracy_list) / len(first_class_5_accuracy_list)
    avg_first_class_6_accuracy = sum(first_class_6_accuracy_list) / len(first_class_6_accuracy_list)
    avg_first_class_7_accuracy = sum(first_class_7_accuracy_list) / len(first_class_7_accuracy_list)
    avg_first_class_8_accuracy = sum(first_class_8_accuracy_list) / len(first_class_8_accuracy_list)
    avg_first_class_9_accuracy = sum(first_class_9_accuracy_list) / len(first_class_9_accuracy_list)
    avg_first_class_10_accuracy = sum(first_class_10_accuracy_list) / len(first_class_10_accuracy_list)
    avg_first_class_11_accuracy = sum(first_class_11_accuracy_list) / len(first_class_11_accuracy_list)
    print("shooting初始平均准确率：")
    print(avg_first_class_1_accuracy)
    print("biking初始平均准确率：")
    print(avg_first_class_2_accuracy)
    print("diving初始平均准确率：")
    print(avg_first_class_3_accuracy)
    print("golf初始平均准确率：")
    print(avg_first_class_4_accuracy)
    print("riding初始平均准确率：")
    print(avg_first_class_5_accuracy)
    print("juggle初始平均准确率：")
    print(avg_first_class_6_accuracy)
    print("swing初始平均准确率：")
    print(avg_first_class_7_accuracy)
    print("tennis初始平均准确率：")
    print(avg_first_class_8_accuracy)
    print("jumping初始平均准确率：")
    print(avg_first_class_9_accuracy)
    print("spiking初始平均准确率：")
    print(avg_first_class_10_accuracy)
    print("walk初始平均准确率：")
    print(avg_first_class_11_accuracy)
    print("---------------------------------------------------")
    avg_class_1_accuracy = sum(class_1_accuracy_list) / len(class_1_accuracy_list)
    avg_class_2_accuracy = sum(class_2_accuracy_list) / len(class_2_accuracy_list)
    avg_class_3_accuracy = sum(class_3_accuracy_list) / len(class_3_accuracy_list)
    avg_class_4_accuracy = sum(class_4_accuracy_list) / len(class_4_accuracy_list)
    avg_class_5_accuracy = sum(class_5_accuracy_list) / len(class_5_accuracy_list)
    avg_class_6_accuracy = sum(class_6_accuracy_list) / len(class_6_accuracy_list)
    avg_class_7_accuracy = sum(class_7_accuracy_list) / len(class_7_accuracy_list)
    avg_class_8_accuracy = sum(class_8_accuracy_list) / len(class_8_accuracy_list)
    avg_class_9_accuracy = sum(class_9_accuracy_list) / len(class_9_accuracy_list)
    avg_class_10_accuracy = sum(class_10_accuracy_list) / len(class_10_accuracy_list)
    avg_class_11_accuracy = sum(class_11_accuracy_list) / len(class_11_accuracy_list)
    print("shooting最终平均准确率：")
    print(avg_class_1_accuracy)
    print("biking最终平均准确率：")
    print(avg_class_2_accuracy)
    print("diving最终平均准确率：")
    print(avg_class_3_accuracy)
    print("golf最终平均准确率：")
    print(avg_class_4_accuracy)
    print("riding最终平均准确率：")
    print(avg_class_5_accuracy)
    print("juggle最终平均准确率：")
    print(avg_class_6_accuracy)
    print("swing最终平均准确率：")
    print(avg_class_7_accuracy)
    print("tennis最终平均准确率：")
    print(avg_class_8_accuracy)
    print("jumping最终平均准确率：")
    print(avg_class_9_accuracy)
    print("spiking最终平均准确率：")
    print(avg_class_10_accuracy)
    print("walk最终平均准确率：")
    print(avg_class_11_accuracy)
    print("---------------------------------------------------")
    print("整体初始平均准确率：")
    a = [avg_first_class_1_accuracy, avg_first_class_2_accuracy, avg_first_class_3_accuracy, avg_first_class_4_accuracy,
         avg_first_class_5_accuracy, avg_first_class_6_accuracy, avg_first_class_7_accuracy, avg_first_class_8_accuracy,
         avg_first_class_9_accuracy, avg_first_class_10_accuracy, avg_first_class_11_accuracy]
    print(sum(a) / len(a))
    print(sum(whole_first_accuracy_list) / len(whole_first_accuracy_list))
    print("---------------------------------------------------")
    print("整体平均准确率：")
    b = [avg_class_1_accuracy, avg_class_2_accuracy, avg_class_3_accuracy, avg_class_4_accuracy,
         avg_class_5_accuracy, avg_class_6_accuracy, avg_class_7_accuracy, avg_class_8_accuracy,
         avg_class_9_accuracy, avg_class_10_accuracy, avg_class_11_accuracy]
    print(sum(b) / len(b))
    print(sum(whole_accuracy_list) / len(whole_accuracy_list))


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    KFold_MFSSEL("/home/sunbite/MFSSEL/features_not_on_spark_for_mfssel/", "/home/sunbite/MFSSEL/model_test/")
    endtime = datetime.datetime.now()
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
    print(
        '-------------KFold_MFSSEL Running time: %s Seconds--------------' % (endtime - starttime).seconds)
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
