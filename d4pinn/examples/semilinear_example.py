"""
例子2：用D4-PINN求解半线性椭圆方程
-Δu + alpha * |u|^2 u = 1, 边界条件u=0
这是你论文的核心问题
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from d4_pinn import D4PINN

# 1. 初始化模型
model = D4PINN(hidden_dim=100, num_layers=5)

# 2. 定义PDE参数
alpha_true = 1.0  # 真实的非线性系数

# 3. 定义PDE残差
def pde_residual(u, x, y):
    # 半线性椭圆方程: -Δu + alpha * |u|^2 u = 1
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    return -u_xx - u_yy + alpha_true * torch.abs(u)**2 * u - 1.0

# 4. 定义边界条件
n_bc = 2048
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

# 5. 一键训练！
print("开始训练半线性椭圆方程...")
model.train(pde_residual, boundary_conditions, epochs=30000)

# 6. 预测
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
u = model.predict(X.flatten(), Y.flatten()).reshape(100, 100)

# 7. 画图
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, u, cmap='jet')
plt.colorbar(label='u(x,y)')
plt.title('D4-PINN Solution of Semilinear Elliptic Equation')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('semilinear_solution.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"训练完成！alpha={alpha_true}")
print("结果已保存为 semilinear_solution.png")
