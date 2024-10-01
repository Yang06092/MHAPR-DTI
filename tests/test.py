import numpy as np
import os
import pandas as pd

# 定义数据集维度
M = 100  # 你可以根据需要修改维度大小
N = 200

# 创建./data_DC/目录，如果它不存在
if not os.path.exists('../data_DC'):
    os.makedirs('../data_DC')

# 生成一个MXN维度的0和1矩阵，随机生成
d_t = np.random.randint(0, 2, size=(M, N))
pd.DataFrame(d_t).to_csv('../data_DC/d_t.csv', index=False, header=False)
print("文件 d_t.csv 已生成并保存。")

# 生成两个MXM维度的矩阵，数据分布在0到1之间的浮点数
d_ss = np.random.rand(M, M)
pd.DataFrame(d_ss).to_csv('../data_DC/d_ss.csv', index=False, header=False)
print("文件 d_ss.csv 已生成并保存。")

# 生成两个NXN维度的矩阵，数据分布在0到1之间的浮点数
p_ss = np.random.rand(N, N)
pd.DataFrame(p_ss).to_csv('../data_DC/p_ss.csv', index=False, header=False)
print("文件 p_ss.csv 已生成并保存。")


