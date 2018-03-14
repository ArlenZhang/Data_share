"""
    desc: matplotlib is perfect for drawing pictures to describe data
    author: LongYinZ
    date: 2018.1.30
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

# section 1 : sin and cosine using default settings
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)
plt.plot(X, C)
plt.plot(X, S)
# plt.show()

# section 2 : 自定义设置
# create a new figure of size 8*6 points, using 100 dots per inch
plt.figure(figsize=(16, 8), dpi=100)

# create a new subplot from a grid of 1*1
plt.subplot(241)
# 这时候再画图将画在第一个子图中
plt.plot(X, C, color="blue", linewidth=1.0, linestyle="-")
# set x limits
plt.xlim(-4.0, 4.0)
plt.ylim(-1.0, 1.0)
# 对于这个图像 我们更关心横坐标对应的最值的x值，插入x_sticks
plt.xticks([-np.pi, 0, np.pi], [r'$-\pi$', r'$0$', r'$+\pi$'])


# create a new subplot
plt.subplot(242)
plt.plot(X, S, color="green", linewidth=1.0, linestyle="-.")
plt.xlim(-4.0, 4.0)
plt.ylim(-1.0, 1.0)

# save all
plt.savefig("graphs/matplotlib1.png", dpi=100)
# plt.show()

# section 3 : 对XY之间关系的一般化理解，绘制一般图像
x = np.linspace(-10, 10, 10000)
y = (x-1)**2
plt.subplot(243)
plt.plot(x, y, color="red", linestyle="-")
# plt.show()

# section 4 : 怎么画出数据点和直线拟合效果
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 3, 2, 4, 5, 7])
plt.subplot(244)
# 上面代表实际数据点的分布情况 在下面用参数bo来显示离散数据
plt.plot(x, y, "bo", color="blue", label="Real Data")
# 给出如下的数据拟合情况
x = np.linspace(1, 6, 1000)
y = x
plt.plot(x, y, color="red", label="Predicted Data")
plt.legend()  # 加入者行代码才能显示标签
# plt.show()

# section 4 : Bar plot 画柱状图
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 3, 2, 4, 5, 7])
plt.subplot(245)
plt.bar(x, y, facecolor="#9999ff", edgecolor="white")
# 怎么将数据标在每个bar上
for x, y in zip(x, y):
    plt.text(x+0.1, y+0.05, '%i' % y, ha='center', va='bottom')

# section 5 : Contour Plots
plt.subplot(246)
def f(x, y):  return(1-x/2+x**5+y**3)*np.exp(-x**2-y**2)
n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap='jet')
C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)

# section 6 : image show
plt.subplot(247)
img_path = "graphs/matplotlib1.png"
image_matrix = mpimg.imread(img_path)
plt.imshow(image_matrix)

# section 7 : 饼状图
plt.subplot(248)
data = np.array([2, 2, 3, 6])
labels = np.array(['a', 'b', 'c', 'd'])
plt.pie(data, labels=labels)

# section 8 : 3D视图
fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
plt.show()

