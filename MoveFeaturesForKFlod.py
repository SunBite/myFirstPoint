# 作用：
#   合并并两个目录/文件夹。
#   将目录A合并到目录B，同级目录下，
#   将A中有，B中没有的目录完全复制到B中；
#   将A中没有，B中有的目录不做改动；
#   将A中做了修改的文件，在B的同级目录下创建一个副本。（注意不是覆盖）
#
# 适应场景：
#   一般A是从B复制过来的文件夹，做了修改后，想合并回B。
#   这样在A中做的一些改动我们就不知道了，这个程序的作用是，
#   将A中修改的部分在B中更新。
#
# 版本：
#   将修改后的文件完全复制过去
#   旧的文件创建副本
#   基于文件的MD5值判断是否修改过
#

import os
import shutil
import time
import hashlib
import sys


def Help():  # 输出帮助文档
    print("""  
PathMerge.py
作者:freecode
创建时间：2016.4.9 20:15
作用：
    合并并两个目录/文件夹。
    将目录A合并到目录B，同级目录下，
    将A中有，B中没有的目录完全复制到B中；
    将A中没有，B中有的目录不做改动；
    将A中做了修改的文件，在B的同级目录下创建一个副本。（注意不是覆盖）

适应场景：
    一般A是从B复制过来的文件夹，做了修改后，想合并回B。
    这样在A中做的一些改动我们就不知道了，这个程序的作用是，
    将A中修改的部分在B中更新。

版本：
    将修改后的文件完全复制过去
    旧的文件创建副本
    基于文件的MD5值判断是否修改过
""")


def GetFileMd5(filename):  # 计算文件的md5值
    if not os.path.isfile(filename):
        return
    myhash = hashlib.md5()
    f = open(filename, 'rb')
    while True:
        b = f.read(8096)
        if not b:
            break
        myhash.update(b)
    f.close()
    return myhash.hexdigest()


def isModify(A_file, B_file):  # 判断两个文件是否相同，如果不同，表示修改过
    # 参数需是绝对路径
    return GetFileMd5(A_file) != GetFileMd5(B_file)


def Stamp2Time(Stamp):  # 将时间戳转换成时间显示格式
    timeArray = time.localtime(Stamp)
    Time = time.strftime("%Y年%m月%d日 %H时%M分%S秒 旧文件副本", timeArray)
    return Time


def Merge(A_path, B_path):  # 合并两个目录
    B_paths = os.listdir(B_path)  # 获取当前B中的目录结构
    for fp in os.listdir(A_path):  # 遍历当前A目录中的文件或文件夹
        A_new_path = os.path.join(A_path, fp)  # A中的文件或目录
        B_new_path = os.path.join(B_path, fp)  # B中对应的文件或路径，不一定存在

        if os.path.isdir(A_new_path):  # A中的目录
            if os.path.exists(B_new_path):  # 如果在B中存在
                Merge(A_new_path, B_new_path)  # 继续合并下一级目录
            else:  # 如果在B中不存在
                print('[目录]\t%s ===> %s' % (A_new_path, B_new_path))
                shutil.copytree(A_new_path, B_new_path)  # 完全复制目录到B

        elif os.path.isfile(A_new_path):  # A中的文件
            if os.path.exists(B_new_path):  # 如果在B中存在
                s = os.stat(B_new_path)
                if isModify(A_new_path, B_new_path) == True:  # 如果该文件修改过
                    # 创建副本
                    suffix = B_new_path.split('.')[-1]  # 得到文件的后缀名
                    # 将B中原文件创建副本
                    B_copy_path = B_new_path[:-len(suffix) - 1] + "(%s)." % (Stamp2Time(s.st_mtime)) + suffix
                    print('[副本]\t%s ===> %s' % (A_new_path, B_copy_path))
                    shutil.copy2(B_new_path, B_copy_path)
                    # 将A中修改后文件复制过来
                    print('[文件]\t%s ===> %s' % (A_new_path, B_new_path))
                    shutil.copy2(A_new_path, B_new_path)
                else:  # 如果该文件没有修改过
                    pass  # 不复制

            else:  # 如果在B中不存在
                # 将该文件复制过去
                print('[文件]\t%s ===> %s' % (A_new_path, B_new_path))
                shutil.copy2(A_new_path, B_new_path)


# 运行模式
if __name__ == '__main__':
    classnames = ["basketball", "biking", "diving", "golf_swing", "horse_riding",
                  "soccer_juggling", "swing", "tennis_swing", "trampoline_jumping",
                  "volleyball_spiking", "walking"]
    testandtrains = ["test", "train"]
    featureDirNames = ["_81feature/", "_81videoFeature/",
                       "_30feature/", "_30videoFeature/",
                       "hogfeature", "hogvideoFeature"
                       ]
    featruePath = "/home/sunbite/MFSSEL/features_not_on_spark_backup/"
    for classname in classnames:
        for featureDirName in featureDirNames:
            path1 = featruePath + classname + os.sep + "test" + os.sep + featureDirName
            path2 = featruePath + classname + os.sep + "train" + os.sep + featureDirName
            Merge(path1, path2)
            shutil.copytree(path2, "/home/sunbite/MFSSEL/features_not_on_spark_for_mfssel/" + classname + os.sep + featureDirName )
#     print("""
#         欢迎使用PathMerge！
#         本程序将会把目录A合并到目录B，即 A ===> B
#         将A目录中修改的内容在B目录中更新
#         合并规则具体见 PathMerge.Help()
#         """)
#     if len(sys.argv) == 1:
#         path1 = input('请输入A目录：').strip()
#         path2 = input('请输入B目录：').strip()
#     elif len(sys.argv) == 2:
#         path1 = sys.argv[1].strip()
#         print('A目录为：%s\n' % (path1))
#         path2 = input('请输入B目录：').strip()
#     elif len(sys.argv) == 3:
#         path1 = sys.argv[1].strip()
#         print('A目录为：%s\n' % (path1))
#         path2 = sys.argv[2].strip()
#         print('B目录为：%s\n' % (path2))
#     else:
#         print('ERROR：参数错误!\n参数最多有三个!\n')
#         input('\n请按回车键(Enter)退出……')
#         sys.exit(0)
#     # 去除目录的引号
#     if path1[0] == '\"':
#         path1 = path1[1:-1]
#     if path2[0] == '\"':
#         path2 = path2[1:-1]
#
#     print("""
# 开始合并目录 %s
# 　　　到目录 %s
# %s ===> %s
# """ % (path1, path2, path1, path2))
#
#     try:
#         print('合并中……')
#         Merge(path1, path2)
#         print('')
#     except Exception as e:
#         print('合并失败！')
#         print('失败原因：\n', e)
#     else:
#         print('合并成功！')
#
#     input('\n请按回车键(Enter)退出……')
