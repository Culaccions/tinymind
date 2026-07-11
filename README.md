# tinymind

从零开始实现的小参数量大语言模型——powered by minimind  
minimind模型架构：



![](img/image.png)





![](img/LLM-structure-moe.jpg)



RMSNorm定义:  
假设输入向量为 $x=(x_1,x_2,...,x_n)$,RMSNorm 的计算分为两步：

1. 计算 $x_i^2$ 的平方和 $S_i=\sum_{j=1}^n x_j^2$
2. 计算 $x_i$ 的归一化值 $x_i'=\frac{x_i}{\sqrt{\frac{S_i}{n}}+\epsilon}\gamma$  
   其中 $\epsilon$ 是一个小的常量，用于避免除零错误,$\gamma$ 是缩放因子,是一个可训练的参数，默认值为 $1.0$。

