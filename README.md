# MLStudy 

了解信息时代最基本的几个概念，这些概念构成了整个信息时代的基石也是整个信息时代的核心。

#### 1. 信息量的衡量指标：信息熵  
**H(x)表示用以消除这个事件的不确定性所需要的统计信息量即为信息熵。**

离散型信息熵的公式：  
变量取值为i到n，所有取值的熵值之和）
![](http://latex.codecogs.com/gif.latex?\\frac{\\partial J}{\\partial \\theta_k^{(j)}}=\\sum_{i:r(i,j)=1}{\\big((\\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\\big)x_k^{(i)}}+\\lambda \\xtheta_k^{(j)})

```math
   H(X) = - \sum_{x = i}^n P(x_i)\log P(x_i)
``` 

连续型信息熵的公式：  
P(x)：概率密度函数
```math
   H(x) = - \intop_x P(x)\log (P(x))dx
```

对应的Python代码：
```python
  import numpy as num
  def ComputerEntropy(x, probability):
      proba
```
