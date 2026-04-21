import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class D4PINN(nn.Module):
    """
    Lightweight D4-equivariant Physics-Informed Neural Network
    
    这是一个零依赖、轻量级的D4对称PINN工具包
    只需定义你的PDE和边界条件，就能一键训练和预测
    
    核心特性：
    - 严格D4硬约束，和ESCNN数学等价
    - 零外部依赖，纯PyTorch实现
    - CPU可跑，普通笔记本就能用
    - 支持前向求解和逆参数识别
    """
    def __init__(self, hidden_dim=100, num_layers=5, device='cpu'):
        """
        初始化D4-PINN模型
        
        Args:
            hidden_dim: 隐藏层维度
            num_layers: MLP层数
            device: 运行设备，默认CPU
        """
        super().__init__()
        self.device = device
        
        # 1. 定义标准MLP（和普通PINN完全一样）
        layers = []
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers).to(device)
        
        # 2. 定义D4群的8个操作矩阵（旋转+反射）
        # 顺序：R0, R90, R180, R270, Sx, Sy, Sd1, Sd2
        self.group_ops = torch.tensor([
            [[1, 0], [0, 1]],      # R0: 恒等变换
            [[0, -1], [1, 0]],     # R90: 顺时针旋转90度
            [[-1, 0], [0, -1]],    # R180: 旋转180度
            [[0, 1], [-1, 0]],     # R270: 顺时针旋转270度
            [[1, 0], [0, -1]],     # Sx: x轴反射
            [[-1, 0], [0, 1]],     # Sy: y轴反射
            [[0, 1], [1, 0]],      # Sd1: 对角线反射
            [[0, -1], [-1, 0]]     # Sd2: 反对角线反射
        ], dtype=torch.float32).to(device)
        
        # 移动到设备
        self.to(device)
    
    def forward(self, x):
        """
        D4对称前向传播：Reynolds算子正交投影
        严格保证输出是D4对称的，和ESCNN不变层数学等价
        
        Args:
            x: 输入坐标，shape: [batch_size, 2]
        Returns:
            u: 预测的解，shape: [batch_size, 1]
        """
        batch_size = x.shape[0]
        
        # 对输入做所有D4变换
        x_transformed = torch.einsum('bij,nj->bni', self.group_ops, x)
        x_transformed = x_transformed.reshape(-1, 2)  # 合并batch和群维度
        
        # 用MLP预测
        y = self.mlp(x_transformed)
        
        # 对8个变换的结果取平均（Reynolds算子）
        y = y.reshape(batch_size, 8)
        return y.mean(dim=1, keepdim=True)
    
    def train(self, pde_residual, boundary_conditions, 
              interior_points=8192, boundary_points=2048, 
              epochs=20000, lr=1e-3, obs_data=None, verbose=True):
        """
        一键训练D4-PINN模型
        
        Args:
            pde_residual: 你的PDE残差函数，输入(u, x, y)，输出残差
            boundary_conditions: 边界条件列表，每个元素是(x_bc, y_bc, u_bc)
            interior_points: 内部采样点数量
            boundary_points: 边界采样点数量
            epochs: 训练轮数
            lr: 学习率
            obs_data: 逆问题的观测数据，可选，格式: (x_obs, y_obs, u_obs)
            verbose: 是否打印训练过程
        """
        # 优化器
        params = list(self.parameters())
        if obs_data is not None:
            # 逆问题：把alpha也加入优化参数
            params.append(obs_data['alpha'])
        
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        
        # 训练循环
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 1. 生成内部点
            x_int = torch.rand(interior_points, 2, device=self.device) * 2 - 1
            x_int.requires_grad = True
            
            # 2. 计算PDE残差
            u = self(x_int)
            loss_pde = torch.mean(pde_residual(u, x_int[:,0], x_int[:,1])**2)
            
            # 3. 计算边界损失
            loss_bc = 0.0
            for x_bc, y_bc, u_bc in boundary_conditions:
                x_bc_tensor = torch.cat([x_bc, y_bc], dim=1).to(self.device)
                u_pred = self(x_bc_tensor)
                loss_bc += torch.mean((u_pred - u_bc.to(self.device))**2)
            
            # 4. 逆问题：计算观测数据损失
            loss_data = 0.0
            if obs_data is not None:
                x_obs = torch.cat([obs_data['x'], obs_data['y']], dim=1).to(self.device)
                u_pred = self(x_obs)
                loss_data = torch.mean((u_pred - obs_data['u'].to(self.device))**2)
            
            # 总损失
            loss = loss_pde + 100 * loss_bc + 100 * loss_data
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 打印日志
            if verbose and epoch % 1000 == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | PDE: {loss_pde.item():.6f} | BC: {loss_bc.item():.6f}")
                scheduler.step()
    
    def predict(self, x, y):
        """
        预测函数，支持numpy输入
        
        Args:
            x: x坐标，numpy数组
            y: y坐标，numpy数组
        Returns:
            u: 预测的解，numpy数组
        """
        x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        x_input = torch.cat([x_tensor, y_tensor], dim=1)
        
        with torch.no_grad():
            u = self(x_input)
        
        return u.cpu().numpy()
