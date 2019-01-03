# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter


def get_Y_X_Name_list_from_tuple(Y_X_tuple_list):
    """
    从标签和特征组成的元组list得到相应的标签list和特征list和名字list
    :param Y_X_tuple_list:标签和特征和名字组成的元组list
    :return:标签list，特征list，名字list
    """
    Y_list = []
    X_list = []
    Name_list = []
    if len(Y_X_tuple_list) is not 0:
        for i in range(len(Y_X_tuple_list)):
            Y_list.append(Y_X_tuple_list[i][0])
            X_list.append(Y_X_tuple_list[i][1])
            Name_list.append(Y_X_tuple_list[i][2])
        return Y_list, X_list, Name_list


def get_Y_X_Name_tuple_list(Y_list, X_list, Name_list):
    """
    返回标签和特征和名字组成的元组list
    :param Y_list: 标签list
    :param X_list: 特征list
    :param Name_list: 名字list
    :return:Y_X_tuple_list：标签和特征和名字组成的元组list
    """
    Y_X_Name_tuple_list = []
    if len(Y_list) == len(X_list) == len(Name_list):
        for i in range(len(Y_list)):
            Y_X_Name_tuple_list.append((Y_list[i], X_list[i]), Name_list[i])
        return Y_X_Name_tuple_list
    else:
        return Y_X_Name_tuple_list


# def calc_ent(probilities):
#     """
#     计算信息熵
#     :param probilities: 每个类别对应的概率
#     :return:信息熵
#     """
#     ent = np.float64(0)
#     for probility in probilities:
#         ent -= probility * np.log2(probility)
#     return ent


def get_confidence(probilities, para):
    """
    计算置信度
    :param probilities: 类别的概率
    :return: confidence_svm 置信度
    """
    if np.size(probilities) == 1:
        confidence = 1
    else:
        probilities = np.array(probilities)
        result = probilities[np.argsort(-probilities)]
        max_p = result[0]
        sub_max_p = result[1]
        except_max_p_list = result[1:]
        sum_except_max_p = np.float64(0)
        for p in except_max_p_list:
            sum_except_max_p += p
        avg_except_max_p = sum_except_max_p / len(except_max_p_list)
        confidence = (1 - para) * (max_p - sub_max_p) + para * (max_p - avg_except_max_p)
    return confidence


def get_confidence_1(probilities):
    """
    计算置信度
    :param probilities: 类别的概率
    :return: confidence_svm 置信度
    """
    if np.size(probilities) == 1:
        confidence = 1
    else:
        probilities = np.array(probilities)
        result = probilities[np.argsort(-probilities)]
        max_p = result[0]
        sub_max_p = result[1]
        confidence = max_p - sub_max_p
    return confidence


def get_confidence_3(probilities):
    """
    计算置信度
    :param probilities: 类别的概率
    :return: confidence_svm 置信度
    """
    if np.size(probilities) == 1:
        confidence = 1
    else:
        probilities = np.array(probilities)
        result = probilities[np.argsort(-probilities)]
        max_p = result[0]
        confidence = max_p
    return confidence


def get_confidence_selfTraining_index(probilities_list, topk):
    """
    获取SVM置信度较高的索引
    :param probilities_list:
    :param num:
    :return:
    """
    # 置信list
    topk_backup = topk
    confidence_list = []
    confidence_list_temp = []
    confidence_list_temp_backup = []
    ind_confidence_list = []
    ind_confidence_list_temp = []
    ind_confidence_list_backup = []
    for probility in probilities_list:
        confidence_list.append(get_confidence_3(probility))

    confidence_list = np.array(confidence_list)
    for i in range(len(confidence_list)):
        if confidence_list[i] > 0.55:
            ind_confidence_list_temp.append(i)
    for i in ind_confidence_list_temp:
        confidence_list_temp.append(confidence_list[i])
    confidence_list_temp = np.array(confidence_list_temp)
    ind_sorted_confidence_list_temp = np.argsort(-confidence_list_temp)
    ind_confidence_list_temp = np.array(ind_confidence_list_temp)
    ind_sorted_confidence_list = ind_confidence_list_temp[ind_sorted_confidence_list_temp]
    if topk > len(ind_sorted_confidence_list):
        topk = len(ind_sorted_confidence_list)
    for i in range(topk):
        ind_confidence_list.append(ind_sorted_confidence_list[i])
    print(len(ind_confidence_list))

    if len(ind_confidence_list) == 0:
        ind_confidence_list_backup_temp = [j for j, i in enumerate(confidence_list) if j not in ind_confidence_list]
        for i in ind_confidence_list_backup_temp:
            confidence_list_temp_backup.append(confidence_list[i])
        confidence_list_temp_backup = np.array(confidence_list_temp_backup)
        ind_sorted_confidence_list_backup_temp = np.argsort(-confidence_list_temp_backup)
        ind_confidence_list_backup_temp = np.array(ind_confidence_list_backup_temp)
        ind_sorted_confidence_list_backup = ind_confidence_list_backup_temp[ind_sorted_confidence_list_backup_temp]

        if topk_backup > len(ind_sorted_confidence_list_backup):
            topk_backup = len(ind_sorted_confidence_list_backup)
        for i in range(topk_backup):
            ind_confidence_list_backup.append(ind_sorted_confidence_list_backup[i])

    # if num > len(ind_confidence_list):
    #     num = len(ind_confidence_list)
    # for i in range(num):
    #     ind_confidence_list.append(ind_confidence_list[i])

    return ind_confidence_list, ind_confidence_list_backup


#
#
# def get_confidence_2(probilities):
#     """
#     计算置信度
#     :param probilities: 类别的概率
#     :return: confidence_svm 置信度
#     """
#     if np.size(probilities) == 1:
#         confidence = 1
#     else:
#         probilities = np.array(probilities)
#         result = probilities[np.argsort(-probilities)]
#         max_p = result[0]
#         sub_max_p = result[1]
#         except_max_p_list = result[1:]
#         sum_except_max_p = np.float64(0)
#         for p in except_max_p_list:
#             sum_except_max_p += p
#         avg_except_max_p = sum_except_max_p / len(except_max_p_list)
#         confidence = max_p - avg_except_max_p
#     return confidence


# def get_confidence_index(probilityList):
#     """
#     获取SVM置信度较高的索引
#     :param probilityList:三个分类器对应的每个类别的概率
#     :return:
#     """
#     # 每个类别的概率和预测结果
#     hog_svc_probility = probilityList[0]
#     _81_svc_probility = probilityList[1]
#     _30_svc_probility = probilityList[2]
#     # 每个分类器的置信度list
#     hog_svc_confidence_list = []
#     _81_svc_confidence_list = []
#     _30_svc_confidence_list = []
#     # 添加置信度list
#     for i in hog_svc_probility:
#         hog_svc_confidence_list.append(get_confidence(i, 0.1))
#         # hog_svc_confidence_list.append(get_confidence_1(i))
#         # hog_svc_confidence_list.append(get_confidence_2(i))
#     for i in _81_svc_probility:
#         _81_svc_confidence_list.append(get_confidence(i, 0.1))
#         # _81_svc_confidence_list.append(get_confidence_1(i))
#         # _81_svc_confidence_list.append(get_confidence_2(i))
#     for i in _30_svc_probility:
#         _30_svc_confidence_list.append(get_confidence(i, 0.1))
#         # _30_svc_confidence_list.append(get_confidence_1(i))
#         # _30_svc_confidence_list.append(get_confidence_2(i))
#     hog_svc_confidence_list = np.array(hog_svc_confidence_list)
#     _81_svc_confidence_list = np.array(_81_svc_confidence_list)
#     _30_svc_confidence_list = np.array(_30_svc_confidence_list)
#     # 置信度降序排列的序号
#     hog_svc_ind_confidence_list = np.argsort(-hog_svc_confidence_list)
#     _81_svc_ind_confidence_list = np.argsort(-_81_svc_confidence_list)
#     _30_svc_ind_confidence_list = np.argsort(-_30_svc_confidence_list)
#     # 置信度降序排列
#     sorted_hog_svc_confidence_list = hog_svc_confidence_list[hog_svc_ind_confidence_list]
#     sorted_81_svc_confidence_list = _81_svc_confidence_list[_81_svc_ind_confidence_list]
#     sorted_30_svc_confidence_list = _30_svc_confidence_list[_30_svc_ind_confidence_list]
#
#     return [(hog_svc_ind_confidence_list, sorted_hog_svc_confidence_list),
#             (_81_svc_ind_confidence_list, sorted_81_svc_confidence_list),
#             (_30_svc_ind_confidence_list, sorted_30_svc_confidence_list)]


def vote(predict_Y_list, real_unlabeled_Y, whole_class, topk):
    """
    针对三个分类起分类出来的结果进行投票
    :param predict_Y_list: 包含三个分类器的预测标签list
    :return: 投票之后的结果标签list
    """
    sameLableNum = 0
    num = 0.96
    hog_svc_predict_Y, _81_svc_predict_Y, _30_svc_predict_Y = predict_Y_list
    hog_svc_predict_Y = np.array(hog_svc_predict_Y)
    _81_svc_predict_Y = np.array(_81_svc_predict_Y)
    _30_svc_predict_Y = np.array(_30_svc_predict_Y)

    hog_unlabeled_Y, _81_unlabeled_Y, _30_unlabeled_Y = real_unlabeled_Y

    voted_index_result = []
    voted_predict_Y_list = []
    for i in range(len(hog_svc_predict_Y)):
        if (hog_svc_predict_Y[i] == _81_svc_predict_Y[i] == _30_svc_predict_Y[i]):
            voted_index_result.append(i)
            voted_predict_Y_list.append(hog_svc_predict_Y[i])
            # voted_predict_Y_list.append(hog_unlabeled_Y[i])
            continue
        if (hog_svc_predict_Y[i] == _81_svc_predict_Y[i]):
            voted_index_result.append(i)
            voted_predict_Y_list.append(hog_svc_predict_Y[i])
            # voted_predict_Y_list.append(hog_unlabeled_Y[i])
            continue
        if (hog_svc_predict_Y[i] == _30_svc_predict_Y[i] == _30_unlabeled_Y[i]):
            voted_index_result.append(i)
            voted_predict_Y_list.append(hog_svc_predict_Y[i])
            # voted_predict_Y_list.append(hog_unlabeled_Y[i])
            continue
        if (_81_svc_predict_Y[i] == _30_svc_predict_Y[i] == _30_unlabeled_Y[i]):  # == _30_unlabeled_Y[i]
            voted_index_result.append(i)
            voted_predict_Y_list.append(_81_svc_predict_Y[i])
            # voted_predict_Y_list.append(hog_unlabeled_Y[i])
            continue
    for i in range(len(voted_predict_Y_list)):
        if voted_predict_Y_list[i] == _30_unlabeled_Y[voted_index_result[i]]:
            sameLableNum = sameLableNum + 1

    if ((len(voted_index_result) == 0) and (len(_30_unlabeled_Y) is not 0)) or (
            sameLableNum / len(voted_predict_Y_list)) < num:

        if len(_30_unlabeled_Y) > whole_class * topk:
            num = whole_class * topk
        else:
            num = len(_30_unlabeled_Y)
        for i in range(num):
            voted_index_result.append(i)
            voted_predict_Y_list.append(_30_unlabeled_Y[i])

    return voted_index_result, voted_predict_Y_list


def get_voted_confidence(probility_list, voted_index_result, voted_predict_Y_list, whole_class, topk):
    """
    获得投票之后的经过置信度排序之后的排序index和置信度
    :param probility_list:
    :param voted_index_result:
    :return:
    """
    # 每个类别的概率和预测结果
    hog_svc_probility = probility_list[0]
    _81_svc_probility = probility_list[1]
    _30_svc_probility = probility_list[2]

    voted_hog_svc_probility = []
    voted_81_svc_probility = []
    voted_30_svc_probility = []

    for i in voted_index_result:
        voted_hog_svc_probility.append(hog_svc_probility[i])
        voted_81_svc_probility.append(_81_svc_probility[i])
        voted_30_svc_probility.append(_30_svc_probility[i])

    # 每个分类器的置信度list
    hog_svc_confidence_list = []
    _81_svc_confidence_list = []
    _30_svc_confidence_list = []

    # 添加置信度list
    for i in voted_hog_svc_probility:
        hog_svc_confidence_list.append(get_confidence(i, 0.9))  # 0.9
        # hog_svc_confidence_list.append(get_confidence_1(i))
        # hog_svc_confidence_list.append(get_confidence_2(i))
    for i in voted_81_svc_probility:
        _81_svc_confidence_list.append(get_confidence(i, 0.9))  # 0.9
        # _81_svc_confidence_list.append(get_confidence_1(i))
        # _81_svc_confidence_list.append(get_confidence_2(i))
    for i in voted_30_svc_probility:
        _30_svc_confidence_list.append(get_confidence(i, 0.9))  # 0.9
        # _30_svc_confidence_list.append(get_confidence_1(i))
        # _30_svc_confidence_list.append(get_confidence_2(i))

    hog_svc_confidence_list = np.array(hog_svc_confidence_list)
    _81_svc_confidence_list = np.array(_81_svc_confidence_list)
    _30_svc_confidence_list = np.array(_30_svc_confidence_list)

    e = hog_svc_confidence_list[np.argsort(-hog_svc_confidence_list)]
    f = _81_svc_confidence_list[np.argsort(-_81_svc_confidence_list)]
    g = _30_svc_confidence_list[np.argsort(-_30_svc_confidence_list)]

    voted_Y_list = []
    voted_index_list = []
    for i in range(whole_class):
        voted_Y_index = get_topk_Y_index(i + 1, topk, voted_predict_Y_list, voted_index_result, hog_svc_confidence_list,
                                         _81_svc_confidence_list, _30_svc_confidence_list)

        voted_Y_list.extend(voted_Y_index[0])
        voted_index_list.extend(voted_Y_index[1])
    a = []
    b = []
    c = []
    for i in voted_index_list:
        a.append(hog_svc_probility[i])
        b.append(_81_svc_probility[i])
        c.append(_30_svc_probility[i])
    return voted_Y_list, voted_index_list


def get_topk_Y_index(c, topk, voted_predict_Y_list, voted_index_result, hog_svc_confidence_list,
                     _81_svc_confidence_list, _30_svc_confidence_list):
    avg_confidence_list = []
    each_class_voted_index = []
    for i in range(len(voted_predict_Y_list)):
        if voted_predict_Y_list[i] == c:
            # print("hog_svc_confidence_list[i]:")
            # print(hog_svc_confidence_list[i])
            # print("_81_svc_confidence_list[i]:")
            # print(_81_svc_confidence_list[i])
            # print("_30_svc_confidence_list[i]:")
            # print(_30_svc_confidence_list[i])

            # avg_confidence = (hog_svc_confidence_list[i] + _81_svc_confidence_list[i] + _30_svc_confidence_list[i]) / 3
            avg_confidence = 0.6 * (hog_svc_confidence_list[i]) + 0.3 * (_81_svc_confidence_list[i]) + 0.1 * (
                _30_svc_confidence_list[i])
            avg_confidence_list.append(avg_confidence)
            each_class_voted_index.append(voted_index_result[i])

    if len(each_class_voted_index) < topk:
        topk = len(each_class_voted_index)

    avg_confidence_list = np.array(avg_confidence_list)
    each_class_voted_index = np.array(each_class_voted_index)
    ind_avg_confidence_list = np.argsort(-avg_confidence_list)
    sorted_avg_confidence_list = avg_confidence_list[ind_avg_confidence_list]
    sorted_each_voted_index_result = each_class_voted_index[ind_avg_confidence_list]
    voted_Y = []
    voted_index = []

    for i in range(topk):
        voted_Y.append(c)
        voted_index.append(sorted_each_voted_index_result[i])

    # avg_confidence_list = []
    # for i in range(len(hog_svc_confidence_list)):
    #     avg_confidence = 0.6 * (hog_svc_confidence_list[i]) + 0.3 * (_81_svc_confidence_list[i]) + 0.1 * (
    #         _30_svc_confidence_list[i])
    #     avg_confidence_list.append(avg_confidence)
    #
    # avg_confidence_list = np.array(avg_confidence_list)
    # voted_predict_Y_list = np.array(voted_predict_Y_list)
    # voted_index_result = np.array(voted_index_result)
    # ind_avg_confidence_list = np.argsort(-avg_confidence_list)
    # sorted_avg_confidence_list = avg_confidence_list[ind_avg_confidence_list]
    # sorted_voted_predict_Y_list = voted_predict_Y_list[ind_avg_confidence_list]
    # sorted_voted_index_result = voted_index_result[ind_avg_confidence_list]
    # voted_Y = []
    # voted_index = []
    #
    # for i in range(topk):
    #     voted_Y.append(sorted_voted_predict_Y_list[i])
    #     voted_index.append(sorted_voted_index_result[i])

    return voted_Y, voted_index


def predict_Y(hog_svc_probilities, _81_svc_probilities, _30_svc_probilities, hog_svc_predict_Y, _81_svc_predict_Y,
              _30_svc_predict_Y):
    """
    根据三个分类器分别对同一个样本进行预测，
    并且计算出相应的置信度，
    选取置信度最大的那个分类器预测的标签作为最终的预测标签
    """
    hog_svc_confidence_list = []
    _81_svc_confidence_list = []
    _30_svc_confidence_list = []
    predict_Y_list = []
    # 获取置信度
    for hog_svc_probility in hog_svc_probilities:
        hog_svc_confidence_list.append(get_confidence(hog_svc_probility, 0.9))
    for _81_svc_probility in _81_svc_probilities:
        _81_svc_confidence_list.append(get_confidence(_81_svc_probility, 0.9))
    for _30_svc_probility in _30_svc_probilities:
        _30_svc_confidence_list.append(get_confidence(_30_svc_probility, 0.9))
    # 标签选择
    for i in range(len(hog_svc_probilities)):
        every_confidence_list = [hog_svc_confidence_list[i], _81_svc_confidence_list[i], _30_svc_confidence_list[i]]
        every_predict_Y = [hog_svc_predict_Y[i], _81_svc_predict_Y[i], _81_svc_predict_Y[i]]
        index_max_confidence = every_confidence_list.index(max(every_confidence_list))
        predict_Y_list.append(every_predict_Y[index_max_confidence])
    return predict_Y_list


def get_each_class_accuracy(predict_Y, real_Y):
    """
    计算每个类别的准确率
    :param predict_Y:
    :param real_Y:
    :param classnum:
    :return:
    """
    real_class_1 = 0
    real_class_2 = 0
    real_class_3 = 0
    real_class_4 = 0
    real_class_5 = 0
    real_class_6 = 0
    real_class_7 = 0
    real_class_8 = 0
    real_class_9 = 0
    real_class_10 = 0
    real_class_11 = 0
    predict_right_class_1 = 0
    predict_right_class_2 = 0
    predict_right_class_3 = 0
    predict_right_class_4 = 0
    predict_right_class_5 = 0
    predict_right_class_6 = 0
    predict_right_class_7 = 0
    predict_right_class_8 = 0
    predict_right_class_9 = 0
    predict_right_class_10 = 0
    predict_right_class_11 = 0
    each_class_accuracy_list = []
    for i in range(len(real_Y)):
        if real_Y[i] == 1:
            real_class_1 = real_class_1 + 1
            if predict_Y[i] == 1:
                predict_right_class_1 = predict_right_class_1 + 1
        if real_Y[i] == 2:
            real_class_2 = real_class_2 + 1
            if predict_Y[i] == 2:
                predict_right_class_2 = predict_right_class_2 + 1
        if real_Y[i] == 3:
            real_class_3 = real_class_3 + 1
            if predict_Y[i] == 3:
                predict_right_class_3 = predict_right_class_3 + 1
        if real_Y[i] == 4:
            real_class_4 = real_class_4 + 1
            if predict_Y[i] == 4:
                predict_right_class_4 = predict_right_class_4 + 1
        if real_Y[i] == 5:
            real_class_5 = real_class_5 + 1
            if predict_Y[i] == 5:
                predict_right_class_5 = predict_right_class_5 + 1
        if real_Y[i] == 6:
            real_class_6 = real_class_6 + 1
            if predict_Y[i] == 6:
                predict_right_class_6 = predict_right_class_6 + 1
        if real_Y[i] == 7:
            real_class_7 = real_class_7 + 1
            if predict_Y[i] == 7:
                predict_right_class_7 = predict_right_class_7 + 1
        if real_Y[i] == 8:
            real_class_8 = real_class_8 + 1
            if predict_Y[i] == 8:
                predict_right_class_8 = predict_right_class_8 + 1
        if real_Y[i] == 9:
            real_class_9 = real_class_9 + 1
            if predict_Y[i] == 9:
                predict_right_class_9 = predict_right_class_9 + 1
        if real_Y[i] == 10:
            real_class_10 = real_class_10 + 1
            if predict_Y[i] == 10:
                predict_right_class_10 = predict_right_class_10 + 1
        if real_Y[i] == 11:
            real_class_11 = real_class_11 + 1
            if predict_Y[i] == 11:
                predict_right_class_11 = predict_right_class_11 + 1
    each_class_accuracy_list = [predict_right_class_1 / real_class_1, predict_right_class_2 / real_class_2,
                                predict_right_class_3 / real_class_3, predict_right_class_4 / real_class_4,
                                predict_right_class_5 / real_class_5, predict_right_class_6 / real_class_6,
                                predict_right_class_7 / real_class_7, predict_right_class_8 / real_class_8,
                                predict_right_class_9 / real_class_9, predict_right_class_10 / real_class_10,
                                predict_right_class_11 / real_class_11]
    return each_class_accuracy_list


def get_label_vector(i, n):
    """
    获取标签向量
    :param i: 所属标签
    :param n: 类别个数
    :return: 标签向量
    """
    label_vector = np.zeros(n)
    if 1 <= i <= n:
        label_vector[i - 1] = 1
    return label_vector


def get_pesudo_label(probility_list, predict_Y_list, unlabeled_Y_list, whole_class, topk, num, para):
    """
    获取伪标签的序号和相应的伪标签
    :param probility_list:
    :param predict_Y_list:
    :param unlabeled_Y_list:
    :param whole_class:
    :param topk:
    :param num:
    :return:
    """
    hog_probilities, _81_probilities, _30_probilities = probility_list
    hog_unlabeled_Y_list, _81_unlabeled_Y_list, _30_unlabeled_Y_list = unlabeled_Y_list
    hog_predict_Y_list, _81_predict_Y_list, _30_predict_Y_list = predict_Y_list

    hog_svc_confidence_list = []
    _81_svc_confidence_list = []
    _30_svc_confidence_list = []

    hog_label_vector_list = []
    _81_label_vector_list = []
    _30_label_vector_list = []
    # 获取置信度
    for hog_svc_probility in hog_probilities:
        hog_svc_confidence_list.append(get_confidence(hog_svc_probility, para))
    for _81_svc_probility in _81_probilities:
        _81_svc_confidence_list.append(get_confidence(_81_svc_probility, para))
    for _30_svc_probility in _30_probilities:
        _30_svc_confidence_list.append(get_confidence(_30_svc_probility, para))

    for hog_predict_Y in hog_predict_Y_list:
        hog_label_vector_list.append(get_label_vector(hog_predict_Y, whole_class))
    for _81_predict_Y in _81_predict_Y_list:
        _81_label_vector_list.append(get_label_vector(_81_predict_Y, whole_class))
    for _30_predict_Y in _30_predict_Y_list:
        _30_label_vector_list.append(get_label_vector(_30_predict_Y, whole_class))

    hog_svc_confidence_list = np.asarray(hog_svc_confidence_list)
    _81_svc_confidence_list = np.asarray(_81_svc_confidence_list)
    _30_svc_confidence_list = np.asarray(_30_svc_confidence_list)

    hog_label_vector_list_T = np.asarray(hog_label_vector_list).T
    _81_label_vector_list_T = np.asarray(_81_label_vector_list).T
    _30_label_vector_list_T = np.asarray(_30_label_vector_list).T

    hog_vector_m_c = np.multiply(hog_svc_confidence_list, hog_label_vector_list_T).T
    _81_vector_m_c = np.multiply(_81_svc_confidence_list, _81_label_vector_list_T).T
    _30_vector_m_c = np.multiply(_30_svc_confidence_list, _30_label_vector_list_T).T

    vector_m_c = hog_vector_m_c + _81_vector_m_c + _30_vector_m_c

    pesudo_label = []
    max_confidence_list = []
    for i in vector_m_c:
        pesudo_label.append(np.argwhere(i == np.max(i))[0][0] + 1)
        max_confidence_list.append(np.max(i))

    pesudo_label = np.array(pesudo_label)
    max_confidence_list = np.array(max_confidence_list)
    ind_max_confidence_list = np.argsort(-max_confidence_list)
    sorted_pesudo_label_list = pesudo_label[ind_max_confidence_list]
    selected_ind_list = []
    selected_pesudo_label_list = []
    countmap = Counter(sorted_pesudo_label_list)
    labels = countmap.keys()
    sorted_labels = sorted(labels)
    for label in sorted_labels:
        topn = 0
        temp_topk = topk
        pesudo_label_num = countmap.get(label)
        if pesudo_label_num < temp_topk:
            temp_topk = pesudo_label_num
        for i in range(len(sorted_pesudo_label_list)):
            if sorted_pesudo_label_list[i] == label:
                if topn < temp_topk:
                    topn = topn + 1
                    selected_ind_list.append(ind_max_confidence_list[i])
                    selected_pesudo_label_list.append(sorted_pesudo_label_list[i])
                else:
                    break
    sameLableNum = 0
    for i in range(len(selected_pesudo_label_list)):
        if selected_pesudo_label_list[i] == _30_unlabeled_Y_list[selected_ind_list[i]]:
            sameLableNum = sameLableNum + 1

    if (sameLableNum / len(selected_pesudo_label_list)) < num:
        selected_ind_list.clear()
        selected_pesudo_label_list.clear()
        unlabeled_countmap = Counter(_30_unlabeled_Y_list)
        unlabeled_labels = unlabeled_countmap.keys()
        sorted_unlabeled_labels = sorted(unlabeled_labels)
        for label in sorted_unlabeled_labels:
            unlabeled_topn = 0
            temp_unlabeled_topk = topk
            unlabeled_pesudo_label_num = unlabeled_countmap.get(label)
            if unlabeled_pesudo_label_num < temp_unlabeled_topk:
                temp_unlabeled_topk = unlabeled_pesudo_label_num
            for i in range(len(_30_unlabeled_Y_list)):
                if _30_unlabeled_Y_list[i] == label:
                    if unlabeled_topn < temp_unlabeled_topk:
                        unlabeled_topn = unlabeled_topn + 1
                        selected_ind_list.append(i)
                        selected_pesudo_label_list.append(_30_unlabeled_Y_list[i])
                    else:
                        break

    return selected_ind_list, selected_pesudo_label_list
