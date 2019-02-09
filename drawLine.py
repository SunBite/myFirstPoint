# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
fontSet = FontProperties(fname=r"/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")

# x = [220, 440, 660, 880, 1100, 1320]
# mfssel = [61.805555555555557, 64.583333333333343, 66.666666666666657, 69.444444444444443, 79.861111111111114, 81.25]
# coknnsvm = []

# x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# y = [78.5555555556, 78.7777777778, 79.1666666667, 79.5833333333, 79.7777777778, 80.1111111111, 80.5555555556,
#      80.1666666667, 79.8888888889]
# plt.plot(x, y, marker='o', label='不同置信度参数下的准确率变化', mec='r', mfc='w')
# plt.xlabel('置信度参数', fontproperties=fontSet)
# plt.ylabel('准确率', fontproperties=fontSet)
# plt.legend(prop=zhfont)
# plt.show()
name_list = ['文献[23]', '文献[24]', '文献[25]', '本文方法']
num_list = [73.20, 76.06, 78.6, 83.48]
rects=plt.bar(range(len(num_list)), num_list, color='rgby')
# X轴标题
index=[0,1,2,3]
#index=[float(c)+0.4 for c in index]
plt.ylim(ymax=90, ymin=0)
plt.xticks(index, name_list,fontproperties=fontSet)
plt.ylabel("准确率", fontproperties=fontSet) #X轴标签
for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height, str(height)+'%', ha='center', va='bottom')
plt.show()
