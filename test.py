import pyhdfs
from skimage import io
import cv2
import numpy as np
fs = pyhdfs.HdfsClient('10.3.11.131:50070')
print(fs.get_home_directory())
print(fs.listdir("/CoKNNSVM/keyframe1"))
response = fs.open("/CoKNNSVM/keyframe1/basketball_v_shooting_01_01_keyFrame_0.jpg")
def bytestounit8codechanger(frame):
    """
    把dtype=bytes的flatten ndarray转换成dtype=unit8的ndarray
    :param frame: dtype=bytes的flatten ndarray
    :return: frame: dtype=unit8的ndarray
    """
    # 把ndarray.bytes转成ndarray.str
    bytestostrdecoder = np.vectorize(lambda x: bytes.decode(x))
    frame = bytestostrdecoder(frame)
    # 把ndarray.str转成ndarray.uint8
    strtouint8decoder = np.vectorize(lambda x: np.uint8(x))
    frame = strtouint8decoder(frame)
    return frame
a = response.read()
b =cv2.imread("/home/sunbite/Co_KNN_SVM_TMP/keyframe/basketball_v_shooting_01_01_keyFrame_0.jpg")
frame = bytestounit8codechanger(response.read())
print(frame)



# strtouint8decoder = np.vectorize(lambda x: np.uint8(x))
# frame = strtouint8decoder(response.read())
# frame = np.reshape(frame, (240, 320, 3))
# cv2.imshow("11",frame)

# class PackageHdfs():
#
#     def __init__(self):
#         self.fs = pyhdfs.HdfsClient('10.3.11.131:50070')
#
#     # 删除
#     def delFile(self, path):
#         fs = self.fs
#         fs.delete(path)
#
#     # 上传文件
#     def upload(self, fileName, tmpFile):
#         fs = self.fs
#         fs.copy_from_local(fileName, tmpFile)
#
#     # 新建目录
#     def makdir(self, filePath):
#         fs = self.fs
#         if not fs.exists(filePath):
#             # os.system('hadoop fs -mkdir '+filePath)
#             fs.mkdirs(filePath)
#             return 'mkdir'
#         return 'exits'
#
#     # 重命名
#     def rename(self, srcPath, dstPath):
#         fs = self.fs
#         if not fs.exists(srcPath):
#             return
#         fs.rename(srcPath, dstPath)
