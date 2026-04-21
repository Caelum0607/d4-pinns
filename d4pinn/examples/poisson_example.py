"""
例子1：用D4-PINN求解泊松方程
-Δu = 1, 边界条件u=0
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from d4_pinn import D4PINN

# 1. 初始化模型
model = D4PINN(hidden_dim=100, num_layers=5)

# 2. 定义PDE残差
def pde_residual(u, x, y):
    # 泊松方程: -Δu = 1
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    return -u_xx - u_yy - 1.0

# 3. 定义边界条件（正方形边界u=0）
n_bc = 2048
# 四个边界：上、下、左、右
x_bc = torch.cat([
    torch.ones(n_bc//4),          # 右边界x=1
    -torch.ones(n_bc//4),         # 左边界x=-1
    torch.rand(n_bc//2)*2-1       # 上下边界x随机
])
y_bc = torch.cat([
    torch.rand(n_bc//2)*2-1,      # 左右边界y随机
    torch.ones(n_bc//4),          # 上边界y=1
    -torch.ones(n_bc//4)          # 下边界y=-1
])
u_bc = torch.zeros(n_bc, 1)
boundary_conditions = [(x_bc, y_bc, u_bc)]

# 4. 一键训练！
print("开始训练泊松方程...")
model.train(pde_residual, boundary_conditions, epochs=20000)

# 5. 预测
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
u = model.predict(X.flatten(), Y.flatten()).reshape(100, 100)

# 6. 画图
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, u, cmap='jet')
plt.colorbar(label='u(x,y)')
plt.title('D4-PINN Solution of Poisson Equation')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('poisson_solution.png', dpi=300, bbox_inches='tight')
plt.show()

print("训练完成！结果已保存为 poisson_solution.png")
