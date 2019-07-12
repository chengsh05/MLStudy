<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# MLStudy
了解信息时代最基本的几个概念，这些概念构成了整个信息时代的基石也是整个信息时代的核心。

## 信息量的衡量指标：信息熵  
**H(x)表示用以消除这个事件的不确定性所需要的统计信息量即为信息熵。**

离散型信息熵的公式：  
变量取值为i到n，所有取值的熵值之和  

$$ H(X) = - \sum_{x = i}^n P(x_i)\log P(x_i)$$  

连续型信息熵的公式：  
P(x)：概率密度函数  
$$ H(x) = - \int_x P(x)\log (P(x))dx $$
对应的Python代码：
```python
import numpy as np
def ComputerEntropy(x, p_x):
      if(isinstance(p_x, float)):
            H_x = -p_x * np.log2(p_x)
            return x, H_x
#Test Case
x1, H_x1 = ComputerEntropy("532", 0.5)
x2, H_x2 = ComputerEntropy("531", 0.2)
x3, H_x3 = ComputerEntropy("530", 0.3)
print(x1, H_x1)
print(x2, H_x2)
print(x3, H_x3)
print("All H(x):", H_x1 + H_x2 + H_x3)
```

## 概率分布的基本概念
#### 概率质量函数与概率密度函数 
概率质量函数和概率密度函数都可以理解为概率函数, 只是一个用于表达离散型取值对应概率的相关函数, 一个是用于表达连续性取值概率分布的相关函数.  

#### PDF：概率密度函数（probability density function）
 在数学中，连续型随机变量的概率密度函数（在不至于混淆时可以简称为密度函数）是一个描述这个随机变量的输出值，在某个确定的取值点附近的可能性的函数。

#### PMF : 概率质量函数（probability mass function)
 在概率论中，概率质量函数是离散随机变量在各特定取值上的概率。

#### CDF : 累积分布函数 (cumulative distribution function)
 概率密度函数的积分，能完整描述一个实随机变量X的概率分布。

#### 期望:
数学期望是随机变量的重要特征之一,随机变量X的数学期望记为E(X),E(X)是X的算术平均的近似值,数学期望表示了X的平均值大小。
- 当X为离散型随机变量时,并且其概率分布函数为\(P(X = x_k) = P_k\), 其中k = 1,2,...,n;则数学期望为:
  $$ E(x) = \sum_{k=1}^n x_k p_k $$
- 当X为连续型随机变量时,设其概率密度为f(x),则数学期望为
  $$E(x)= \int_{-\infty}^{+\infty}xf(x)dx$$
#### 方差:
数学期望给出了随机变量的平均大小,现实生活中我们还经常关心随机变量的取值在均值周围的散布程度,而方差就是这样的一个数字特征
设X是随机变量,并且\(E{[X-E(x)^2]}\)存在,则称它为X的方差,记为D(X)  

- 当X为离散型时: 
$$D(x) = \sum_k[x_k - E(X)]^2*P_k$$  
- 当X为连续型时: 
$$D(x) = \int_{-\infty}^{+\infty}[x_k - E(X)]^2*f(x)dx$$
#### 标准差:
- 方差的算术平方根\(\sigma(X) = \sqrt{D(x)} \)为X的标准差

- 另外,\(D(x) = E{[(X-E(x))^2]}\) 经过化解可得
  $$D(X) = E(X^2) – [E(X)]^2$$ 我们一般计算的时候常用这个式子。

#### 协方差:

#### 相关系数:
#### 常见的概率分布:
各自的函数表示方法, 信息熵的推导与计算, 期望与方差的计算.  
##### 离散型分布
###### 伯努利分布:
###### 二项分布:
###### 多项式分布:
###### 离散型均匀分布:
###### 泊松分布

##### 连续型分布
###### 正态分布(高斯分布):
###### 指数分布:
###### 均匀分布:

###### 贝叶斯公式与理解