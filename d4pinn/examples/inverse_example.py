"""
例子3：逆参数识别（工程应用核心）
从10个稀疏观测点，反演半线性方程的非线性系数alpha
这是你论文的工程价值核心
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from d4_pinn import D4PINN

# 1. 初始化模型
model = D4PINN(hidden_dim=100, num_layers=5)

# 2. 真实参数（我们要反演的就是这个！）
alpha_true = 1.0

# 3. 定义PDE（alpha是未知的！）
alpha = torch.nn.Parameter(torch.tensor(0.5))  # 初始化猜测值
def pde_residual(u, x, y):
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    return -u_xx - u_yy + alpha * torch.abs(u)**2 * u - 1.0

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

# 5. 生成10个稀疏观测点（模拟传感器数据）
print("生成10个稀疏观测点，加5%噪声...")
x_obs = torch.rand(10, 2) * 2 - 1
# 先用真实alpha生成观测数据
with torch.no_grad():
    # 临时用真实alpha跑一次，得到真实的u
    alpha_temp = alpha_true
    def temp_residual(u, x, y):
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        return -u_xx - u_yy + alpha_temp * torch.abs(u)**2 * u - 1.0
    model.train(temp_residual, boundary_conditions, epochs=10000, verbose=False)
    u_obs_true = model(x_obs[:,0], x_obs[:,1]).detach() + 0.05 * torch.randn(10, 1)  # 加5%噪声

# 逆问题的观测数据
obs_data = {
    'x': x_obs[:,0],
    'y': x_obs[:,1],
    'u': u_obs_true,
    'alpha': alpha
}

# 6. 训练！反演alpha
print("开始逆问题训练，反演alpha...")
model.train(pde_residual, boundary_conditions, epochs=30000, obs_data=obs_data)

# 7. 打印结果
error = abs(alpha.item() - alpha_true) * 100
print(f"\n=== 逆问题结果 ===")
print(f"真实alpha: {alpha_true:.4f}")
print(f"预测alpha: {alpha.item():.4f}")
print(f"相对误差: {error:.2f}%")

# 8. 预测解
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
u = model.predict(X.flatten(), Y.flatten()).reshape(100, 100)

# 9. 画图
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, u, cmap='jet')
plt.colorbar(label='u(x,y)')
plt.scatter(x_obs[:,0].numpy(), x_obs[:,1].numpy(), c='white', s=50, marker='x', label='观测点')
plt.legend()
plt.title(f'Inverse Problem: alpha={alpha.item():.4f} (True={alpha_true})')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('inverse_solution.png', dpi=300, bbox_inches='tight')
plt.show()

print("结果已保存为 inverse_solution.png")
