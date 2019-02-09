# -*- coding: utf-8 -*-
from pyspark import SparkConf,SparkContext
import datetime
sc = SparkContext(appName="MFSSELTrainOnSpark")
_81Features = sc.textFile("file:///home/sunbite/MFSSEL/features_on_spark/hogvideoFeature/")
_81Features.count()

# class MFSSELTrainOnSpark:
#
#     def __init__(self, filepath, savepath):
#         """
#         初始化方法
#         :param filepath: hdfs上的要读取的features目录
#         :param savepath: 保存的地址
#         """
#         self.__filepath = filepath
#         self.__savepath = savepath
#         self.__classmap = {"shooting": 1, "biking": 2, "diving": 3, "golf": 4, "riding": 5,
#                            "juggle": 6, "swing": 7, "tennis": 8, "jumping": 9, "spiking": 10, "walk": 11}
#
#     def MFSSELTrainOnSpark(self):
#         """
#         训练模型，预测结果
#         """
#         #_81FeaturesPath = self.__filepath + "_81videoFeature/"
#         sc = SparkContext(appName="MFSSELTrainOnSpark")
#         _81Features = sc.textFile("/home/sunbite/MFSSEL/features_on_spark/_81videoFeature")
#         _81Features.map(lambda x: x.split(" ")).map(lambda x: (x[1], x[2:])).count()
#
#
# if __name__ == '__main__':
#     starttime = datetime.datetime.now()
#     MFSSELTrainOnSpark("/home/sunbite/MFSSEL/features_on_spark/",
#                        '/home/sunbite/MFSSEL/model/MFSSEL.model').MFSSELTrainOnSpark()
#     endtime = datetime.datetime.now()
#     print('----------------------------------------------------------------------------')
#     print('----------------------------------------------------------------------------')
#     print('----------MFSSELTrainOnSpark Running time: %s Seconds-----------' % (endtime - starttime).seconds)
#     print('----------------------------------------------------------------------------')
#     print('----------------------------------------------------------------------------')
