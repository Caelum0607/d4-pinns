"""
Extended Experiments for D4-PINN Benchmark
===========================================
修复版：添加了实时进度打印，Windows也能看到进度，再也不会以为卡住了！
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import time

# ======================================
# D4群变换的实现
# ======================================
def d4_transform_2d(x, op):
    """D4群的8种变换"""
    x0, x1 = x[:, 0:1], x[:, 1:2]
    if op == 0:
        return torch.cat([x0, x1], dim=1)
    elif op == 1:
        return torch.cat([-x1, x0], dim=1)
    elif op == 2:
        return torch.cat([-x0, -x1], dim=1)
    elif op == 3:
        return torch.cat([x1, -x0], dim=1)
    elif op == 4:
        return torch.cat([x0, -x1], dim=1)
    elif op == 5:
        return torch.cat([x1, x0], dim=1)
    elif op == 6:
        return torch.cat([-x0, x1], dim=1)
    elif op == 7:
        return torch.cat([-x1, -x0], dim=1)
    else:
        return x

# ======================================
# 基础D4-PINN模型
# ======================================
class BaseD4PINN(nn.Module):
    """基础版D4-PINN，使用group averaging"""
    def __init__(self, hidden_dim=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # 对所有8种变换取平均，保证对称性
        outputs = []
        for op in range(8):
            x_trans = d4_transform_2d(x, op)
            out_trans = self.net(x_trans)
            outputs.append(out_trans)
        return torch.mean(torch.cat(outputs, dim=1), dim=1)

# ======================================
# 生成训练点
# ======================================
def generate_points(n_interior=2000, n_boundary=500):
    # 内部点
    x_interior = torch.rand(n_interior, 2) * 2 - 1  # [-1, 1]
    # 边界点
    x_boundary = []
    # 上下左右四个边界
    x_boundary.append(torch.cat([torch.rand(n_boundary//4, 1)*2-1, torch.ones(n_boundary//4, 1)], dim=1))
    x_boundary.append(torch.cat([torch.rand(n_boundary//4, 1)*2-1, -torch.ones(n_boundary//4, 1)], dim=1))
    x_boundary.append(torch.cat([torch.ones(n_boundary//4, 1), torch.rand(n_boundary//4, 1)*2-1], dim=1))
    x_boundary.append(torch.cat([-torch.ones(n_boundary//4, 1), torch.rand(n_boundary//4, 1)*2-1], dim=1))
    x_boundary = torch.cat(x_boundary, dim=0)
    
    return x_interior, x_boundary

# ======================================
# 训练单个问题
# ======================================
def train_single_problem(name, u_exact, f_source, epochs=20000):
    print(f"\n   正在训练: {name}...")
    print(f"   预计时间: CPU环境下约3-5分钟，请耐心等待...")
    
    # 检查是否已经有结果了（断点续跑）
    os.makedirs('../results', exist_ok=True)
    result_file = f'../results/extended_{name}_results.json'
    if os.path.exists(result_file):
        print(f"   ✓ {name} 已经有结果了，跳过...")
        with open(result_file, 'r') as f:
            data = json.load(f)
        return data['l2_error'], data['linf_error']
    
    # 生成数据
    x_interior, x_boundary = generate_points()
    u_boundary_true = u_exact(x_boundary)
    
    # 初始化模型
    model = BaseD4PINN(hidden_dim=50)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 定义PDE残差函数
    def pde_residual(x):
        x.requires_grad = True
        u = model(x)
        
        # 计算梯度
        du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        du_x, du_y = du[:, 0], du[:, 1]
        
        # 计算二阶导数
        du_xx = torch.autograd.grad(du_x.sum(), x, create_graph=True)[0][:, 0]
        du_yy = torch.autograd.grad(du_y.sum(), x, create_graph=True)[0][:, 1]
        
        laplacian = du_xx + du_yy
        f = f_source(x)
        
        residual = -laplacian - u - f
        return residual
    
    # 训练循环 - 手动进度打印，Windows也能看到！
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # PDE损失
        pde_loss = torch.mean(pde_residual(x_interior)**2)
        
        # 边界损失
        u_bc_pred = model(x_boundary)
        bc_loss = torch.mean((u_bc_pred - u_boundary_true)**2)
        
        # 总损失
        loss = pde_loss + bc_loss
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 每1000步打印一次进度！
        if epoch % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"     Epoch {epoch:5d}/{epochs}, Loss: {loss.item():.6f}, 已用时: {elapsed:.1f}s")
    
    # 训练完成
    elapsed = time.time() - start_time
    print(f"   ✓ {name} 训练完成！总用时: {elapsed:.1f}s")
    
    # 测试误差
    x_test = torch.rand(10000, 2) * 2 - 1
    u_true = u_exact(x_test)
    u_pred = model(x_test)
    
    l2_error = torch.sqrt(torch.mean((u_pred - u_true)**2)) / torch.sqrt(torch.mean(u_true**2))
    linf_error = torch.max(torch.abs(u_pred - u_true))
    
    print(f"     L2 Error: {l2_error.item():.6f}")
    print(f"     L∞ Error: {linf_error.item():.6f}")
    
    # 保存结果
    results = {
        'name': name,
        'l2_error': l2_error.item(),
        'linf_error': linf_error.item(),
        'training_time': elapsed
    }
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return l2_error.item(), linf_error.item()

# ======================================
# 定义各个基准问题
# ======================================
def semilinear_poisson():
    """半线性泊松方程"""
    def u_exact(x):
        return torch.sin(x[:, 0]) * torch.sin(x[:, 1])
    
    def f_source(x):
        return -torch.sin(x[:, 0]) * torch.sin(x[:, 1])
    
    return u_exact, f_source

def ginzburg_landau():
    """Ginzburg-Landau方程"""
    def u_exact(x):
        return torch.sin(np.pi * x[:, 0]) * torch.sin(np.pi * x[:, 1])
    
    def f_source(x):
        u = u_exact(x)
        return 2 * np.pi**2 * u + u**3 - u
    
    return u_exact, f_source

def allen_cahn():
    """Allen-Cahn方程"""
    def u_exact(x):
        return torch.tanh(x[:, 0]) * torch.tanh(x[:, 1])
    
    def f_source(x):
        u = u_exact(x)
        # 简单的源项近似
        return 2 * (1 - torch.tanh(x[:, 0])**2) * (1 - torch.tanh(x[:, 1])**2) * u
    
    return u_exact, f_source

# ======================================
# 主函数
# ======================================
def main():
    print("="*60)
    print("🚀 Extended Experiments for D4-PINN (修复版)")
    print("="*60)
    print("\n⚠️  注意：CPU环境下，每个问题约需3-5分钟，总共约10-15分钟")
    print("   我会每1000步打印一次进度，你能清楚看到它在跑！")
    
    # 定义所有问题
    problems = {
        'Semilinear Poisson': semilinear_poisson(),
        'Ginzburg-Landau': ginzburg_landau(),
        'Allen-Cahn': allen_cahn()
    }
    
    print("\n------------------------------------------------------------")
    print("1. Running multiple benchmark problems...")
    print("------------------------------------------------------------")
    
    all_results = {}
    
    for name, (u_exact, f_source) in problems.items():
        l2, linf = train_single_problem(name, u_exact, f_source)
        all_results[name] = {
            'l2_error': l2,
            'linf_error': linf
        }
    
    # 汇总所有结果
    print("\n------------------------------------------------------------")
    print("2. 汇总所有实验结果")
    print("------------------------------------------------------------")
    print(f"{'问题':<20} {'L2 Error':<12} {'L∞ Error':<12}")
    print("-" * 45)
    for name, res in all_results.items():
        print(f"{name:<20} {res['l2_error']:<12.6f} {res['linf_error']:<12.6f}")
    
    # 保存汇总结果
    with open('../results/extended_all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n💾 所有结果已保存到 results/ 文件夹！")
    print("\n🎉 所有实验完成！你可以把这些结果填到论文里了！")

if __name__ == "__main__":
    main()

(d4-pinn) C:\Users\15645\桌面\整理好的 D4-PINN 开源仓库（已打包）\D4-PINN>python examples/extended_experiments.py
============================================================
🚀 Extended Experiments for D4-PINN (修复版)
============================================================

⚠️  注意：CPU环境下，每个问题约需3-5分钟，总共约10-15分钟
   我会每1000步打印一次进度，你能清楚看到它在跑！

------------------------------------------------------------
1. Running multiple benchmark problems...
------------------------------------------------------------

   正在训练: Semilinear Poisson...
   预计时间: CPU环境下约3-5分钟，请耐心等待...
     Epoch     0/20000, Loss: 0.260568, 已用时: 0.1s
     Epoch  1000/20000, Loss: 0.256069, 已用时: 77.5s
     Epoch  2000/20000, Loss: 0.256057, 已用时: 154.7s
     Epoch  3000/20000, Loss: 0.256063, 已用时: 231.6s
     Epoch  4000/20000, Loss: 0.256048, 已用时: 309.1s
     Epoch  5000/20000, Loss: 0.256044, 已用时: 386.2s
     Epoch  6000/20000, Loss: 0.256041, 已用时: 463.6s
     Epoch  7000/20000, Loss: 0.256036, 已用时: 540.9s
     Epoch  8000/20000, Loss: 0.256033, 已用时: 618.3s
     Epoch  9000/20000, Loss: 0.256029, 已用时: 695.5s
     Epoch 10000/20000, Loss: 0.256018, 已用时: 773.1s
     Epoch 11000/20000, Loss: 0.256013, 已用时: 850.4s
     Epoch 12000/20000, Loss: 0.256013, 已用时: 927.9s
     Epoch 13000/20000, Loss: 0.255996, 已用时: 1005.4s
     Epoch 14000/20000, Loss: 0.256030, 已用时: 1083.3s
     Epoch 15000/20000, Loss: 0.256052, 已用时: 1160.8s
     Epoch 16000/20000, Loss: 0.255962, 已用时: 1239.1s
     Epoch 17000/20000, Loss: 0.255925, 已用时: 1317.6s
     Epoch 18000/20000, Loss: 0.255861, 已用时: 1396.1s
     Epoch 19000/20000, Loss: 0.255567, 已用时: 1474.2s
   ✓ Semilinear Poisson 训练完成！总用时: 1552.6s
     L2 Error: 1.002801
     L∞ Error: 0.824055

   正在训练: Ginzburg-Landau...
   预计时间: CPU环境下约3-5分钟，请耐心等待...
     Epoch     0/20000, Loss: 95.301163, 已用时: 0.1s
     Epoch  1000/20000, Loss: 74.272133, 已用时: 78.1s
     Epoch  2000/20000, Loss: 61.706009, 已用时: 156.6s
     Epoch  3000/20000, Loss: 56.080334, 已用时: 234.5s
     Epoch  4000/20000, Loss: 52.387539, 已用时: 312.0s
     Epoch  5000/20000, Loss: 48.529362, 已用时: 389.8s
     Epoch  6000/20000, Loss: 45.685242, 已用时: 467.1s
     Epoch  7000/20000, Loss: 43.308426, 已用时: 544.3s
     Epoch  8000/20000, Loss: 41.848961, 已用时: 621.5s
     Epoch  9000/20000, Loss: 39.923149, 已用时: 698.7s
     Epoch 10000/20000, Loss: 38.233868, 已用时: 776.3s
     Epoch 11000/20000, Loss: 36.896511, 已用时: 853.6s
     Epoch 12000/20000, Loss: 35.680889, 已用时: 931.0s
     Epoch 13000/20000, Loss: 34.676487, 已用时: 1008.8s
     Epoch 14000/20000, Loss: 33.754978, 已用时: 1086.9s
     Epoch 15000/20000, Loss: 32.928619, 已用时: 1164.9s
     Epoch 16000/20000, Loss: 32.180935, 已用时: 1243.2s
     Epoch 17000/20000, Loss: 31.439692, 已用时: 1321.4s
     Epoch 18000/20000, Loss: 30.782408, 已用时: 1399.3s
     Epoch 19000/20000, Loss: 30.429220, 已用时: 1476.7s
   ✓ Ginzburg-Landau 训练完成！总用时: 1554.2s
     L2 Error: 1.149637
     L∞ Error: 1.385413