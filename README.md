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
            E = -p_x * np.log2(p_x)
            return x, E
#Test Case
x1, E1 = ComputerEntropy("532", 0.5)
x2, E2 = ComputerEntropy("531", 0.2)
x3, E3 = ComputerEntropy("530", 0.3)
print(x1, E1)
print(x2, E2)
print(x3, E3)
print("All H(x):", E1 + E2 + E3)
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
#### 方差:
#### 标准差:

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
