from Task1 import Dual_SVM
import numpy as np
from cvxopt import solvers

solvers.options['show_progress'] = False

# 钓鱼岛坐标
FishingIsoland = [123.28,25.45]
FishingIsoland = np.array(FishingIsoland)[None, :]

# 仅仅使用沿海城市

China_coastal = [119.28,26.08,#福州
                 121.31,25.03,#台北
                 121.47,31.23,#上海
                 118.06,24.27,#厦门
                 121.46,39.04,#大连
                 122.10,37.50,#威海
                 124.23,40.07]#丹东

Japan_coastal = [129.87,32.75,#长崎
                 130.33,31.36,#鹿儿岛
                 131.42,31.91,#宫崎
                 130.24,33.35,#福冈
                 133.33,15.43,#鸟取
                 138.38,34.98,#静冈
                 140.47,36.37]#水户  

# 添加内陆城市

China = [119.28,26.08,#福州
         121.31,25.03,#台北
         121.47,31.23,#上海
         118.06,24.27,#厦门
         113.53,29.58,#武汉
         104.06,30.67,#成都
         116.25,39.54,#北京
         121.46,39.04,#大连
         122.10,37.50,#威海
         124.23,40.07]#丹东

Japan = [129.87,32.75,#长崎
         130.33,31.36,#鹿儿岛
         131.42,31.91,#宫崎
         130.24,33.35,#福冈
         136.54,35.10,#名古屋
         132.27,34.24,#广岛
         139.46,35.42,#东京
         133.33,15.43,#鸟取
         138.38,34.98,#静冈
         140.47,36.37]#水户

# 只看沿海城市的情况
x = np.concatenate((China_coastal, Japan_coastal), axis=0).reshape(-1, 2)
y = np.concatenate((np.ones((len(China_coastal)//2, 1)), 
                    -np.ones((len(Japan_coastal)//2, 1))), axis=0)

dual_svm = Dual_SVM(x, y)
dual_svm.quadratic_programming()
print("中国标签为+1, 日本为标签为-1.")
print(np.sign(np.insert(FishingIsoland,0,1,axis=1) @ dual_svm.w))

# 加入内陆城市
x = np.concatenate((China, Japan), axis=0).reshape(-1, 2)
y = np.concatenate((np.ones((len(China)//2, 1)), 
                    -np.ones((len(Japan)//2, 1))), axis=0)
print("-------------添加内陆城市后-----------")
dual_svm = Dual_SVM(x, y)
alpha = dual_svm.quadratic_programming()
idx = alpha > 1e-6
print(idx)
print("True对应的内陆城市为支撑向量")
print(np.sign(np.insert(FishingIsoland,0,1,axis=1) @ dual_svm.w))
print("中国标签为+1, 日本为标签为-1.")
