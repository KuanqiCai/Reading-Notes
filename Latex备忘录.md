# 字体相关

- 加粗：

# 上下标置于正上/下方

- 使用`\limits`：
命令格式：`expr1\limits_{expr2}^{expr3}`
举例：$\sum\limits_{i=0}^n {x_i}$
- 如果expr1不是数学符号，则用`\mathop{}`将其转成数学符号
举例：$\mathop{max}\limits_{\pi}^{\theta}$

# 括号相关

- 圆括号

  `\big(\big)`:$\big(\big)$  <=对比=> ()



# 等号对齐

$$
\begin{align*}
  A &= B + C \\
    &= C + D + C \\
    &= 2C + D
\end{align*}
$$

# 各种符号

| latex              | 符号                 | latex  | 符号     |
| ------------------ | -------------------- | ------ | -------- |
| \displaystyle \int | $\displaystyle \int$ | \prod  | $\prod$  |
| \Delta             | $\Delta$             | \nabla | $\nabla$ |
| \wedge            | $\wedge$             |   \Lambda    |    $\Lambda$      |
|                    |                      | \mathbb{R} | $\mathbb{R}$ |
|                    |                      |        |          |
|                    |                      |        |          |
|                    |                      |        |          |
|                    |                      |        |          |
|                    |                      |        |          |

# 矩阵行列式：

$\left |\begin{array}{cccc}
e_1 & e_2 & e_3 \\
a_1 & a_2 & a_3  \\
b_1 & b_2 & b_3 \\
\end{array}\right|$

```
\left |\begin{array}{cccc}
e_1 & e_2 & e_3 \\
a_1 & a_2 & a_3  \\
b_1 & b_2 & b_3 \\
\end{array}\right|
```

将上面的|改成 [],()就可以变成矩阵矩阵。

比如：

$\left [\begin{array}{cccc}
e_1 & e_2 & e_3 \\
a_1 & a_2 & a_3  \\
b_1 & b_2 & b_3 \\
\end{array}\right]$

# 方程组

$$
\begin{cases}
3x + 5y +  z \\
7x - 2y + 4z \\
-6x + 3y + 2z
\end{cases}
$$

```
\begin{cases}
3x + 5y +  z \\
7x - 2y + 4z \\
-6x + 3y + 2z
\end{cases}
```

