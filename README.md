# tinymind

从零开始实现的小参数量大语言模型——powered by minimind  
**minimind模型架构：**



![](img/image.png)





![](img/LLM-structure-moe.jpg)



**RMSNorm定义:**  
假设输入向量为 $x=(x_1,x_2,...,x_n)$,RMSNorm 的计算分为两步：

1. 计算 $x_i^2$ 的平方和 $S_i=\sum_{j=1}^n x_j^2$
2. 计算 $x_i$ 的归一化值 $x_i'=\frac{x_i}{\sqrt{\frac{S_i}{n}}+\epsilon}\gamma$  
   其中 $\epsilon$ 是一个小的常量，用于避免除零错误，$\gamma$是缩放因子,是一个可训练的参数，默认值为 $1.0$。

**ROPE旋转位置编码：**

首先我们先要了解在不改变模长的情况下向量如何进行选择，我们可以使用旋转矩阵

$$
R(\theta)=\begin{pmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{pmatrix}
$$

那么选择位置编码的思想就是，如果当前词为第$m$个词，那就将它旋转$m\theta$度，如果为第$n$个词，则将它旋转$n\theta$度，这里的$\theta$是一个角度的基本单位，是一个常量。

那么我们领旋转后的$q$和$k$为$q'$和$k'$，其中$q'=R(m\theta).q$，$k'=R(n\theta).k$

我们知道$(AB)^T=B^TA^T$那么新的注意力分数就为：

$$
\mathit{Score} = \left(q'\right)^T \cdot k' 
= q^T \cdot R(m\theta)^T \cdot R(n\theta) \cdot k
$$

我们将$R(m\theta)^T.R(n\theta)$展开得到：$R(\\alpha) = \\begin{pmatrix} \\cos\\alpha & -\\sin\\alpha \\\\ \\sin\\alpha & \\cos\\alpha \\end{pmatrix}$

$$
R(m\theta)^\mathrm{T}
=
\begin{pmatrix}
\cos m\theta & \sin m\theta \\
-\sin m\theta & \cos m\theta
\end{pmatrix},\quad
R(n\theta)
=
\begin{pmatrix}
\cos n\theta & -\sin n\theta \\
\sin n\theta & \cos n\theta
\end{pmatrix}
$$

其相乘后每个位置元素计算如下，结合积化和差公式可得：

$$
\begin{aligned}
C_{11}
&= a_{11}b_{11} + a_{12}b_{21} \\
&= \cos m\theta \cdot \cos n\theta + \sin m\theta \cdot \sin n\theta \\
&= \cos\left(n\theta - m\theta\right) \\
&= \cos\big((n-m)\theta\big)
\end{aligned}
$$

$$
\begin{aligned}
C_{12}
&= a_{11}b_{12} + a_{12}b_{22} \\
&= \cos m\theta \cdot (-\sin n\theta) + \sin m\theta \cdot \cos n\theta \\
&= -\left(\cos m\theta \sin n\theta - \sin m\theta \cos n\theta\right) \\
&= -\sin\left(n\theta - m\theta\right) \\
&= -\sin\big((n-m)\theta\big)
\end{aligned}
$$

$$
\begin{aligned}
C_{21}
&= a_{21}b_{11} + a_{22}b_{21} \\
&= -\sin m\theta \cdot \cos n\theta + \cos m\theta \cdot \sin n\theta \\
&= \sin\left(n\theta - m\theta\right) \\
&= \sin\big((n-m)\theta\big)
\end{aligned}
$$

$$
\begin{aligned}
C_{22}
&= a_{21}b_{12} + a_{22}b_{22} \\
&= -\sin m\theta \cdot (-\sin n\theta) + \cos m\theta \cdot \cos n\theta \\
&= \cos m\theta \cos n\theta + \sin m\theta \sin n\theta \\
&= \cos\left(n\theta - m\theta\right) \\
&= \cos\big((n-m)\theta\big)
\end{aligned}
$$

最终结果为：

$$
R(m\theta)^\mathrm{T} R(n\theta)
=
\begin{pmatrix}
\cos\left((n-m)\theta\right) & -\sin\left((n-m)\theta\right) \\
\sin\left((n-m)\theta\right) & \cos\left((n-m)\theta\right)
\end{pmatrix}
= R\big((n-m)\theta\big)
$$

我们发现他会等于$R((n-m)\theta)$，这样我们就得到了关于相对位置的信息，最终注意力分数就可以化简为：

$$
\mathit{Score} = \left(q'\right)^T \cdot k' 
= q^T \cdot R(m\theta)^T \cdot R(n\theta) \cdot k=q^T.R((n-m)\theta).k
$$

上面的例子中$q$和$k$都是二维的，但是在实际中词向量的维度往往是很大的，所以RoPE采用了分治法的思想，两两位一组，每组各转各的：

![](img/rope%E5%88%86%E6%B2%BB.png)

此时出现了一个问题，当我们逆时针旋转$10$度和我们逆时针旋转$370$其实没有本质的区别，那为了区分，RoPE为每一组设计了不同的$\theta$让每一组的转速都不同，组的维数越低，频率越快，组的维数越高，频率越慢。其中$\theta_i = 10000^{-\frac{2(i-1)}{d_{\mathrm{model}}}} i \in \left[1,2,\dots,\frac{d}{2}\right]$这部分借鉴了《Attention is All you need》中正余弦位置编码。

引入不同频率之后，位置信息就由这$n$组不同的旋转向量共同表示了，即使前面的部分有重叠，但是仍然可以靠后面的向量区分出来。

**YARN：**

![](img/YARN%E5%A4%84%E7%90%86%E6%96%B9%E6%B3%95.png)

在原始 RoPE 中，位置 $m$ 对应的旋转角度为 $m⋅θ_i$。其**波长（Wavelength）** $λ_i$​ 定义为该维度完成一次完整旋转（2π）所需的 token 距离：

$$
\lambda_i = \frac{2\pi}{\theta_i}=2\pi.{10000^{\frac{2(i-1)}{d_{\mathrm{model}}}}}
$$

定义比率$r_i$ 来衡量波长相对于训练长度的倍数：

$$
r_i=\frac{L}{\lambda_i}
$$

- 当 $r_i≪1$（即 $λ_i​≫L$）时，属于**低频维度**（波长极长），负责全局绝对位置。
- 当 $r_i≫1$（即 $λ_i​≪L$）时，属于**高频维度**（波长极短），负责局部相对位置。

接着我们求出$i'$这里我们令$i'=i-1$

$$
r_i = \frac{L}{2\pi.{10000^{\frac{2(i-1)}{d_{\mathrm{model}}}}}}
$$

$$
{10000^{\frac{2(i-1)}{d_{\mathrm{model}}}}}=\frac{L}{2\pi.r_i}
$$

$$
e^{\frac{2(i-1)}{d_{model}}.ln1000}=\frac{L}{2\pi.r_i}
$$

$$
\frac{2(i-1)}{d_{model}}=\frac{ln(\frac{L}{2\pi.r_i})}{ln1000}
$$

$$
i'=\frac{d_{model}ln(\frac{L}{2\pi.r_i})}{2.ln1000}
$$

接着计算缩放因子：

$$
\gamma_i=
\begin{cases}
0, & i \le \mathrm{low} \\
\dfrac{i-\mathrm{low}}{\mathrm{high}-\mathrm{low}}, & \mathrm{low} < i < \mathrm{high} \\
1, & i \ge \mathrm{high}
\end{cases}
$$

**GQA:**

