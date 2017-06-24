
# coding: utf-8

# # 1 矩阵运算
# 
# ## 1.1 创建一个 4*4 的单位矩阵

# In[33]:


# 这个项目设计来帮你熟悉 python list 和线性代数
# 你不能调用任何python库，包括NumPy，来完成作业

A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

#TODO 创建一个 4*4 单位矩阵
I = [[1,0,0,0], 
     [0,1,0,0],
     [0,0,1,0],
     [0,0,0,1]]


# ## 1.2 返回矩阵的行数和列数

# In[34]:


# TODO 返回矩阵的行数和列数
def shape(M):
    return len(M),len(M[0])


# ## 1.3 每个元素四舍五入到特定小数数位

# In[35]:


# TODO 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound(M, decPts=4):
    for i in range(len(M)):
        for j in range(len(M[0])):
            M[i][j] = round(M[i][j],decPts)


# ## 1.4 计算矩阵的转置

# In[36]:


# TODO 计算矩阵的转置
def transpose(M):
    _, cols = shape(M)
    MT = [[row[col] for row in M] for col in range(cols)]
    return MT


# ## 1.5 计算矩阵乘法 AB

# In[37]:


# TODO 计算矩阵乘法 AB，如果无法相乘则返回None
def matxMultiply(A, B):
    if len(A[0]) != len(B):
        return None
    BT = transpose(B)
    result = [[sum((a * b) for a, b in zip(row_a, col_b)) for col_b in BT] for row_a in A]
    return result


# ## 1.6 测试你的函数是否实现正确

# **提示：** 你可以用`from pprint import pprint`来更漂亮的打印数据，详见[用法示例](http://cn-static.udacity.com/mlnd/images/pprint.png)和[文档说明](https://docs.python.org/2/library/pprint.html#pprint.pprint)。

# In[38]:


import pprint
pp = pprint.PrettyPrinter(indent = 1, width = 20)


# In[39]:


#TODO 测试1.2 返回矩阵的行和列
print("-" * 10 + " Test 1 " + "-" * 10)

print("矩阵 B：")
pp.pprint(B)

print("返回矩阵 B 的行和列：")
print(shape(B))

#TODO 测试1.3 每个元素四舍五入到特定小数数位
print("-" * 10 + " Test 2 " + "-" * 10)

c = [[1.11111,2.222222,3.333333,4.44444444,5.5555555555,6.66666666666]]

print("矩阵 c：")
pp.pprint(c)

print("四舍五入到小数点 4 位的矩阵 c：")
matxRound(c)
pp.pprint(c)

#TODO 测试1.4 计算矩阵的转置
print("-" * 10 + " Test 3 " + "-" * 10)

print("矩阵 B 的转置：")
BT = transpose(B)
pp.pprint(BT)


#TODO 测试1.5 计算矩阵乘法AB，AB无法相乘
print("-" * 10 + " Test 4 " + "-" * 10)

print("计算无法相乘的两个矩阵的结果：")
p = matxMultiply(B, A)
pp.pprint(p)

#TODO 测试1.5 计算矩阵乘法AB，AB可以相乘
print("-" * 10 + " Test 5 " + "-" * 10)

print("计算可以相乘的两个矩阵的结果：")
q = matxMultiply(A, B)
pp.pprint(q)


# # 2 Gaussign Jordan 消元法
# 
# ## 2.1 构造增广矩阵
# 
# $ A = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n}\\
#     a_{21}    & a_{22} & ... & a_{2n}\\
#     a_{31}    & a_{22} & ... & a_{3n}\\
#     ...    & ... & ... & ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn}\\
# \end{bmatrix} , b = \begin{bmatrix}
#     b_{1}  \\
#     b_{2}  \\
#     b_{3}  \\
#     ...    \\
#     b_{n}  \\
# \end{bmatrix}$
# 
# 返回 $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[40]:


# TODO 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
    return [row_a + row_b for row_a, row_b in zip(A, b)]


# ## 2.2 初等行变换
# - 交换两行
# - 把某行乘以一个非零常数
# - 把某行加上另一行的若干倍：

# In[41]:


# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]

# TODO r1 <--- r1 * scale， scale!=0
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    if scale == 0:
        raise ValueError('scale CANNOT be 0')
    M[r] = [e * scale for e in M[r]]

# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale): 
    M[r1] = [e1 + e2 * scale for e1, e2 in zip(M[r1], M[r2])]


# ## 2.3  Gaussian Jordan 消元法求解 Ax = b

# ### 提示：
# 
# 步骤1 检查A，b是否行数相同
# 
# 步骤2 构造增广矩阵Ab
# 
# 步骤3 逐列转换Ab为化简行阶梯形矩阵 [中文维基链接](https://zh.wikipedia.org/wiki/%E9%98%B6%E6%A2%AF%E5%BD%A2%E7%9F%A9%E9%98%B5#.E5.8C.96.E7.AE.80.E5.90.8E.E7.9A.84-.7Bzh-hans:.E8.A1.8C.3B_zh-hant:.E5.88.97.3B.7D-.E9.98.B6.E6.A2.AF.E5.BD.A2.E7.9F.A9.E9.98.B5)
#     
#     对于Ab的每一列（最后一列除外）
#         当前列为列c
#         寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
#         如果绝对值最大值为0
#             那么A为奇异矩阵，返回None （请在问题2.4中证明该命题）
#         否则
#             使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c） 
#             使用第二个行变换，将列c的对角线元素缩放为1
#             多次使用第三个行变换，将列c的其他元素消为0
#             
# 步骤4 返回Ab的最后一列
# 
# ### 注：
# 我们并没有按照常规方法先把矩阵转化为行阶梯形矩阵，再转换为化简行阶梯形矩阵，而是一步到位。如果你熟悉常规方法的话，可以思考一下两者的等价性。

# In[42]:


# TODO 实现 Gaussain Jordan 方法求解 Ax = b

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16
        
    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""

def gj_Solve(A, b, decPts = 4, epsilon = 1.0e-16):
    #步骤1 检查A，b是否行数相同
    if len(A) != len(b):
        raise ValueError('The number of rows in A and b MUST be the SAME')
    
    #步骤2 构造增广矩阵Ab
    Ab = augmentMatrix(A, b)
    
    #步骤3 逐列转换Ab为化简行阶梯形矩阵

    rows, cols = shape(Ab)
    # TODO
    for c in range(cols-1):
        
        #转置后迭代更加方便
        AbT = transpose(Ab)
        col = AbT[c]
        
        #列c(转置后的行c)中，c~N行（转置后的c~N列），每一个元素的绝对值列表
        abs_cn = [abs(e) for e in col[c:]]
        
        #绝对值列表的最大值
        max_abs = max(abs_cn)
        max_index = abs_cn.index(max_abs) + c
        #max_value = col[max_index]
        #因为精度的问题，所以绝对值最大值小于 epsilon 就看作是等于 0
        if max_abs <= epsilon:
            return None

        #最大值所在行（转置后的列）
        #max_index = abs_cn.index(max_value) + c
        
        #使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c）
        swapRows(Ab, c, max_index)
        
        #使用第二个行变换，将列c的对角线元素缩放为1
        scale_c = 1.0 / Ab[c][c]
        scaleRow(Ab, c, scale_c)
        
        #多次使用第三个行变换，将列c的其他元素消为0
        for i in range(len(A)):
            if Ab[i][c] != 0 and i != c:
                addScaledRow(Ab, i, c, -Ab[i][c])
    
    matxRound(Ab)            
    #步骤4 返回Ab的最后一列
    return [[e_last] for e_last in transpose(Ab)[-1]]


# In[43]:


# c = [1,-3,0]
# abs_cn = [abs(e) for e in c[:]]
# print(abs_cn)
# max_value = max(abs_cn)
# print(max_value)
# max_index = abs_cn[:].index(3) + 0
# print(max_index)


# ## 2.4 证明下面的命题：
# 
# **如果方阵 A 可以被分为4个部分: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} , \text{其中 I 为单位矩阵，Z 为全0矩阵，Y 的第一列全0}$，
# 
# **那么A为奇异矩阵。**
# 
# 提示：从多种角度都可以完成证明
# - 考虑矩阵 Y 和 矩阵 A 的秩
# - 考虑矩阵 Y 和 矩阵 A 的行列式
# - 考虑矩阵 A 的某一列是其他列的线性组合

# TODO 请使用 latex （请参照题目的 latex 写法学习）
# 
# TODO 证明：
# 
# $ |A|= |I||Y| - |Z||X| $
# 
# $\text{其中 Z 为全 0 矩阵，Y 的第一列为 0，所以：}$
# 
# $ |Y| = 0  \\
# |Z| = 0$
# 
# $ \text{所以}  |A| = 0$
# 
# **所以 A 为奇异矩阵。**

# ## 2.5 测试 gj_Solve() 实现是否正确

# In[44]:


# TODO 构造 矩阵A，列向量b，其中 A 为奇异矩阵
A = [[1,-3,1],[-3,9,1],[0,1,-1]]
b = [[4],[0],[1]] 
print augmentMatrix(A, b)
# TODO 构造 矩阵A，列向量b，其中 A 为非奇异矩阵
# TODO 求解 x 使得 Ax = b
# TODO 计算 Ax
# TODO 比较 Ax 与 b
A = [[1,-3,1],[-3,9,1],[0,1,-1]]
b = [[4],[0],[1]]

x = gj_Solve(A, b)  #[13,4,3]

print(x)
if matxMultiply(A,x) == b:
    print ('Ax == b the calculation is right')

C = [[-1,1,1],[1,-4,4],[7,-5,-11]]
#d = [-2,21,0]
d= [[-2],[21],[0]]
y =gj_Solve(C, d)  #[1,-3,2]

print(y)

if matxMultiply(C,y) == d:
    print ('Cy == d the calculation is right')


# # 3 线性回归: 
# 
# ## 3.1 计算损失函数相对于参数的导数 (两个3.1 选做其一)
# 
# 我们定义损失函数 E ：
# $$
# E = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# 证明：
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{其中 }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 请使用 latex （参照题目的 latex写法学习）
# 
# TODO 证明：
# 
# $$
# X^TXh = \begin{bmatrix}
#     x_1 & x_2 & ... & x_n \\
#     1 & 1 & ... & 1\\
# \end{bmatrix}  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix} \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}= \begin{bmatrix}
#      \sum_{i=1}^{n}\limits{x_i^2} & \sum_{i=1}^{n}\limits{x_i}\\
#     \sum_{i=1}^{n}\limits{x_i} & n\\
# \end{bmatrix} \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix} = \begin{bmatrix}
#      \sum_{i=1}^{n}\limits{x_i(mx_i+b)} \\
#      \sum_{i=1}^{n}\limits{mx_i+b}\\
# \end{bmatrix} $$
# 
# $$
# X^TY = \begin{bmatrix}
#     x_1 & x_2 & ... & x_n \\
#     1 & 1 & ... & 1\\
# \end{bmatrix} \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix} = \begin{bmatrix}
#     \sum_{i=1}^{n}\limits{x_iy_i} \\
#     \sum_{i=1}^{n}\limits{y_i} \\
# \end{bmatrix}
# $$
# 
# $$
# 2X^TXh-2X^TY = 2\begin{bmatrix}
#      \sum_{i=1}^{n}\limits{x_i(mx_i+b)} \\
#      \sum_{i=1}^{n}\limits{mx_i+b}\\
# \end{bmatrix}-2\begin{bmatrix}
#     \sum_{i=1}^{n}\limits{x_iy_i} \\
#     \sum_{i=1}^{n}\limits{y_i} \\
# \end{bmatrix} = \begin{bmatrix}
#      \sum_{i=1}^{n}\limits{-2x_i(y_i-mx_i-b)} \\
#      \sum_{i=1}^{n}\limits{-2(y_i-mx_i-b}\\
# \end{bmatrix} 
# $$
# 
# $$
# \text{损失函数为： }
# E = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# $$
# \text{两边对 m 求导： }\frac{\partial E}{\partial m} = {-2x_1(y_1 - mx_1 - b)+(-2x_2(y_2 - mx_2 - b)) + ... + (-2x_n(y_n - mx_n - b))} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \text{两边对 b 求导： }\frac{\partial E}{\partial b} = {-2(y_1 - mx_1 - b) + (-2(y_2 - mx_2 - b)) + ... + (-2(y_n - mx_n - b)) } = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 
# $$
# \text{可以得到： }\begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = \begin{bmatrix}
#      \sum_{i=1}^{n}\limits{-2x_i(y_i-mx_i-b)} \\
#      \sum_{i=1}^{n}\limits{-2(y_i-mx_i-b}\\
# \end{bmatrix}
# $$
# 
# $$
# \text{所以： }\begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$

# ## 3.1 计算损失函数相对于参数的导数（两个3.1 选做其一）
# 
# 证明：
# 
# $$
# E = Y^TY -2(Xh)^TY + (Xh)^TXh
# $$ 
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix}  = \frac{\partial E}{\partial h} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{其中 }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 请使用 latex （请参照题目的 latex 写法学习）
# 
# TODO 证明：

# ## 3.2  线性回归
# 
# ### 求解方程 $X^TXh = X^TY $, 计算线性回归的最佳参数 h

# In[45]:


# TODO 实现线性回归
'''
参数：(x,y) 二元组列表
返回：m，b
'''
def linearRegression(points):
    rows,_ = shape(points)
    X = []
    Y = []
    for i in range(rows):
        X.append([points[i][0], 1])
        Y.append([points[i][1]])

    XT = transpose(X)
    XT_X = matxMultiply(XT, X)
    XT_Y = matxMultiply(XT, Y)
    
    h = gj_Solve(XT_X, XT_Y)
    return h


# ## 3.3 测试你的线性回归实现

# In[46]:


# TODO 构造线性函数

# y = 2 * x + 3
h_test = [[2], [3]]

# TODO 构造 100 个线性函数上的点，加上适当的高斯噪音
import random

points = []

for i in range(100):
    x = random.randint(0, 200)
    y = 2 * x + 3 + random.gauss(0,0.1)
    points.append((x,y))
    

#TODO 对这100个点进行线性回归，将线性回归得到的函数和原线性函数比较

#print(points)
h_new = linearRegression(points)  #h_test = [[2], [3]]
print(h_new)
loss = ((h_new[0][0] - h_test[0][0]) ** 2 + (h_new[1][0] - h_test[1][0]) ** 2) / 2
print(loss < 0.1)


# ## 4.1 单元测试
# 
# 请确保你的实现通过了以下所有单元测试。

# In[47]:


import unittest
import numpy as np

from decimal import *

class LinearRegressionTestCase(unittest.TestCase):
    """Test for linear regression project"""

    def test_shape(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.randint(low=-10,high=10,size=(r,c))
            self.assertEqual(shape(matrix.tolist()),(r,c))


    def test_matxRound(self):

        for decpts in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            dec_true = [[Decimal(str(round(num,decpts))) for num in row] for row in mat]

            matxRound(mat,decpts)
            dec_test = [[Decimal(str(num)) for num in row] for row in mat]

            res = Decimal('0')
            for i in range(len(mat)):
                for j in range(len(mat[0])):
                    res += dec_test[i][j].compare_total(dec_true[i][j])

            self.assertEqual(res,Decimal('0'))


    def test_transpose(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            t = np.array(transpose(mat))

            self.assertEqual(t.shape,(c,r))
            self.assertTrue((matrix.T == t).all())


    def test_matxMultiply(self):

        for _ in range(10):
            r,d,c = np.random.randint(low=1,high=25,size=3)
            mat1 = np.random.randint(low=-10,high=10,size=(r,d)) 
            mat2 = np.random.randint(low=-5,high=5,size=(d,c)) 
            dotProduct = np.dot(mat1,mat2)

            dp = np.array(matxMultiply(mat1,mat2))

            self.assertTrue((dotProduct == dp).all())


    def test_augmentMatrix(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            A = np.random.randint(low=-10,high=10,size=(r,c))
            b = np.random.randint(low=-10,high=10,size=(r,1))

            Ab = np.array(augmentMatrix(A.tolist(),b.tolist()))
            ab = np.hstack((A,b))

            self.assertTrue((Ab == ab).all())

    def test_swapRows(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1, r2 = np.random.randint(0,r, size = 2)
            swapRows(mat,r1,r2)

            matrix[[r1,r2]] = matrix[[r2,r1]]

            self.assertTrue((matrix == np.array(mat)).all())

    def test_scaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            rr = np.random.randint(0,r)
            with self.assertRaises(ValueError):
                scaleRow(mat,rr,0)

            scale = np.random.randint(low=1,high=10)
            scaleRow(mat,rr,scale)
            matrix[rr] *= scale

            self.assertTrue((matrix == np.array(mat)).all())
    
    def test_addScaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1,r2 = np.random.randint(0,r,size=2)

            scale = np.random.randint(low=1,high=10)
            addScaledRow(mat,r1,r2,scale)
            matrix[r1] += scale * matrix[r2]

            self.assertTrue((matrix == np.array(mat)).all())


    def test_gj_Solve(self):

        for _ in range(10):
            r = np.random.randint(low=3,high=10)
            A = np.random.randint(low=-10,high=10,size=(r,r))
            b = np.arange(r).reshape((r,1))

            x = gj_Solve(A.tolist(),b.tolist())
            if np.linalg.matrix_rank(A) < r:
                self.assertEqual(x,None)
            else:
                # Ax = matxMultiply(A.tolist(),x)
                Ax = np.dot(A,np.array(x))
                loss = np.mean((Ax - b)**2)
                # print Ax
                # print loss
                self.assertTrue(loss<0.1)


suite = unittest.TestLoader().loadTestsFromTestCase(LinearRegressionTestCase)
unittest.TextTestRunner(verbosity=3).run(suite)


# In[ ]:




