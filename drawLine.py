# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl

zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')

x = [220, 440, 660, 880, 1100, 1320]
mfssel = [61.805555555555557, 64.583333333333343, 66.666666666666657, 69.444444444444443, 79.861111111111114, 81.25]
coknnsvm = []
plt.plot(x, mfssel, marker='o', label='本文方法', mec='r', mfc='w')
plt.legend(prop=zhfont)
plt.show()
