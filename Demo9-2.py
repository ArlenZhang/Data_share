import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 序列数据
data1 = pd.Series(np.random.randn(1000), index=np.arange(1000))
data1 = data1.cumsum()
data1.plot()

# 矩阵数据
data2 = pd.DataFrame(np.random.randn(1000, 4), index=np.arange(1000), columns=list("ABCD"))
data2 = data2.cumsum()
data2.plot()

# 分布点
ax = data2.plot.scatter(x='A', y='B', color='Blue', label='class_a')
data2.plot.scatter(x='A', y='D', color='Red', label='class_b', ax=ax)

plt.show()
