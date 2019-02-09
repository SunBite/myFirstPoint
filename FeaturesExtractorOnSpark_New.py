# -*- coding: utf-8 -*-
from pyspark import SparkConf, SparkContext, StorageLevel
import numpy as np
import GetFeatures
import os
import FeaturesExtractorOnSpark_New as feos
import cv2
import datetime
import shutil
import sys


class FeaturesExtractorOnSpark:

    def __init__(self, keyframepath, featuressavepath, resizeheight=240, resizewidth=320):
        """
        初始化方法
        :param keyframepath: hdfs上的keyframe信息的路径
        :param featuressavepath: hdfs上的features保存的路径
        :param resizeheight: 重置视频的大小的高
        :param resizewidth: 重置视频的大小的宽
        """
        self.__keyframepath = keyframepath
        self.__featuressavepath = featuressavepath
        self.__resizeheight = resizeheight
        self.__resizewidth = resizewidth

    def featuresextractor(self):
        """
        提取特征
        """
        sc = SparkContext(appName="FeaturesExtractorOnSpark" + os.path.basename(self.__featuressavepath))
        framenameandvaluerdd = sc.textFile(self.__keyframepath)

        def get_frame(line):
            framenamepath = line
            print(framenamepath)
            framebasenamelist = os.path.basename(framenamepath).split("_")

            framename = os.path.dirname(framenamepath) + os.sep + "_".join(framebasenamelist[:-1])
            framenum = float(framebasenamelist[-1].split(".")[0])
            frame = cv2.imread(framenamepath)
            # reshape
            frame = np.reshape(frame, (self.__resizeheight, self.__resizewidth, 3))
            return framename, frame, framenum

        def extractfeatures_81(line):
            """
            读取关键帧路径，进行特征提取
            :param line: hdfs上的一行数据，关键帧的路径
            :return: 关键帧的名字+“ ”+该关键帧的特征
            """
            framename = line[0]
            frame = line[1]
            framenum = line[2]
            # 获取不同组合的特征
            _81feature = GetFeatures._81feature(frame)
            # 把帧编号加入到featurelist里
            _81feature.insert(0, framenum)

            return framename, _81feature

        def extractfeatures_30(line):
            """
            读取关键帧路径，进行特征提取
            :param line: hdfs上的一行数据，关键帧的路径
            :return: 关键帧的名字+“ ”+该关键帧的特征
            """
            framename = line[0]
            frame = line[1]
            framenum = line[2]
            # 获取不同组合的特征
            _30feature = GetFeatures._30feature(frame)

            # 把帧编号加入到featurelist里
            _30feature.insert(0, framenum)

            return framename, _30feature

        def extractfeatures_hog(line):
            """
            读取关键帧路径，进行特征提取
            :param line: hdfs上的一行数据，关键帧的路径
            :return: 关键帧的名字+“ ”+该关键帧的特征
            """
            framename = line[0]
            frame = line[1]
            framenum = line[2]
            # 获取不同组合的特征
            hogfeature = GetFeatures.hog_feature(frame)

            # 把帧编号加入到featurelist里
            hogfeature = list(hogfeature)
            hogfeature.insert(0, framenum)

            return framename, hogfeature

        def asbytes(s):
            if isinstance(s, bytes):
                return s
            return str(s).encode('latin1')

        def asstr(s):
            if isinstance(s, bytes):
                return s.decode('latin1')
            return str(s)

        def makeFeaturesItems(X, fmt='%.18e', delimiter=' '):

            # Py3 conversions first
            if isinstance(fmt, bytes):
                fmt = asstr(fmt)
            delimiter = asstr(delimiter)

            try:
                new_X = ""
                X = np.asarray(X)
                # Handle 1-dimensional arrays
                if X.ndim == 1:
                    # Common case -- 1d array of numbers
                    if X.dtype.names is None:
                        X = np.atleast_2d(X).T
                        ncol = 1

                    # Complex dtype -- each field indicates a separate column
                    else:
                        ncol = len(X.dtype.descr)
                else:
                    ncol = X.shape[1]

                iscomplex_X = np.iscomplexobj(X)
                # `fmt` can be a string with multiple insertion points or a
                # list of formats.  E.g. '%10.5f\t%10d' or ('%10.5f', '$10d')
                if type(fmt) in (list, tuple):
                    if len(fmt) != ncol:
                        raise AttributeError('fmt has wrong shape.  %s' % str(fmt))
                    format = asstr(delimiter).join(map(asstr, fmt))
                elif isinstance(fmt, str):
                    n_fmt_chars = fmt.count('%')
                    error = ValueError('fmt has wrong number of %% formats:  %s' % fmt)
                    if n_fmt_chars == 1:
                        if iscomplex_X:
                            fmt = [' (%s+%sj)' % (fmt, fmt), ] * ncol
                        else:
                            fmt = [fmt, ] * ncol
                        format = delimiter.join(fmt)
                    elif iscomplex_X and n_fmt_chars != (2 * ncol):
                        raise error
                    elif ((not iscomplex_X) and n_fmt_chars != ncol):
                        raise error
                    else:
                        format = fmt
                else:
                    raise ValueError('invalid fmt: %r' % (fmt,))

                for row in X:
                    try:
                        new_X = asbytes(new_X) + asbytes(" ") + asbytes(format % tuple(row))
                    except TypeError:
                        raise TypeError("Mismatch between array dtype ('%s') and "
                                        "format specifier ('%s')"
                                        % (str(X.dtype), format))
                return new_X
            finally:
                pass

        def groupbyframename(line):
            """
            返回关键帧名字和相应的特征
            :param line: 关键帧的名字+“ ”+该关键帧的特征
            :return: myfeatureandname 返回关键帧名字和相应的特征
            """
            # 把帧名字加到list的第一个位置
            framename = line[0]
            ResultIterable = list(line[1])
            for i in range(0, len(ResultIterable)):
                if (ResultIterable[i][0] == float(0)):
                    my_feature_num1 = ResultIterable[i][1:]
                elif (ResultIterable[i][0] == float(1)):
                    my_feature_num2 = ResultIterable[i][1:]
                elif (ResultIterable[i][0] == float(2)):
                    my_feature_num3 = ResultIterable[i][1:]
            # 把3个关键帧的特征结合
            my_feature = my_feature_num1 + my_feature_num2 + my_feature_num3
            my_feature = makeFeaturesItems(my_feature, fmt='%.18f')
            myfeatureandname = asbytes(framename) + my_feature

            # 返回关键帧名字和相应的特征
            return myfeatureandname

        # 分割和提取关键帧
        getFrameRdd = framenameandvaluerdd.map(get_frame)
        extract81FeaturesRdd = getFrameRdd.map(extractfeatures_81).groupByKey().map(groupbyframename)
        extract30FeaturesRdd = getFrameRdd.map(extractfeatures_30).groupByKey().map(groupbyframename)
        extractHogFeaturesRdd = getFrameRdd.map(extractfeatures_hog).groupByKey().map(groupbyframename)
        # 保存到hdfs上
        extract81FeaturesRdd.saveAsTextFile(self.__featuressavepath + "hsvVideoFeature/")
        extract30FeaturesRdd.saveAsTextFile(self.__featuressavepath + "lbpVideoFeature/")
        extractHogFeaturesRdd.saveAsTextFile(self.__featuressavepath + "hogVideoFeature/")
        sc.stop()


if __name__ == '__main__':
    if os.path.exists("/home/sunbite/MFSSEL/features_on_spark/"):
        shutil.rmtree("/home/sunbite/MFSSEL/features_on_spark/")
    starttime = datetime.datetime.now()
    FeaturesExtractorOnSpark(
        r"file:/home/sunbite/MFSSEL/keyframepath_new.txt",
        r"file:/home/sunbite/MFSSEL/features_on_spark/").featuresextractor()
    # feos.FeaturesExtractorOnSpark(
    #     r"hdfs://sunbite-computer:9000/CoKNNSVM/keyframepath1.txt",
    #     r"hdfs://sunbite-computer:9000/CoKNNSVM/features320240-366/").featuresextractor()
    endtime = datetime.datetime.now()
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
    print(
        '-------------FeaturesExtractorOnSpark Running time: %s Seconds--------------' % (endtime - starttime).seconds)
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
