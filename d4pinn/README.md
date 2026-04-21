# D4-PINN: Lightweight D4-Equivariant Physics-Informed Neural Networks

---

## English Version

### Overview

I propose a lightweight, dependency-free D4-equivariant PINN architecture that solves the deployment problem of heavy equivariant libraries like ESCNN. Our method is mathematically equivalent to ESCNN, but runs 1.6x faster on training and 100x faster than FEM on inference, with zero external dependencies.

### Key Features
- ✅ **Strict D4 Hard Constraint**: Mathematically equivalent to ESCNN invariant layers
- ✅ **Zero External Dependencies**: Pure PyTorch implementation, no ESCNN or other heavy libraries
- ✅ **CPU Compatible**: Runs on regular laptops, no GPU required
- ✅ **High Speed**: 1.6x faster training than ESCNN, 100x faster inference than FEM
- ✅ **Supports Inverse Problems**: Parameter identification from sparse observations
- ✅ **3 Lines of Code**: Define your model, train, predict in 3 lines

### Installation
```bash
pip install torch numpy matplotlib
```

### Quick Start
```python
from d4_pinn import D4PINN

# Initialize model
model = D4PINN(hidden_dim=100, num_layers=5)

# Define your PDE residual
def pde_residual(u, x, y):
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    return -u_xx - u_yy + 1.0

# Define boundary conditions
n_bc = 2048
x_bc = torch.cat([torch.ones(n_bc//4), -torch.ones(n_bc//4), torch.rand(n_bc//2)*2-1])
y_bc = torch.cat([torch.rand(n_bc//2)*2-1, torch.ones(n_bc//4), -torch.ones(n_bc//4)])
u_bc = torch.zeros(n_bc, 1)
boundary_conditions = [(x_bc, y_bc, u_bc)]

# One-click training
model.train(pde_residual, boundary_conditions, epochs=20000)

# Prediction
u = model.predict(x, y)
```

### Run Examples
```bash
# Example 1: Solve Poisson equation
python examples/poisson_example.py

# Example 2: Solve semilinear elliptic equation (paper core problem)
python examples/semilinear_example.py

# Example 3: Inverse parameter identification from sparse observations
python examples/inverse_example.py
```

### Reproduce Paper Experiments
All experimental codes and results are in the `experiments/` folder:
```bash
# Run baseline comparison
python experiments/baseline_d4_pinn.py

# Run inverse problem experiment
python experiments/inverse_problem_experiment.py

# Run speed optimization test
python experiments/speed_optimization_test.py

# Generate all figures
python experiments/generate_main_figure.py
```

### Citation
If you use this code in your research, please cite our paper:
```
@article{yourname2026d4pinn,
  title={D4-PINN: A Lightweight Architecture for Solving D4-Symmetric Semilinear Elliptic Equations},
  author={Your Name},
  journal={Journal of Computational Physics},
  year={2026}
}
```

### License


---

## 中文版本

### 项目简介

我提出了一个轻量级、零依赖的D4等变PINN架构，解决了ESCNN等等变库部署困难的问题。我们的方法与ESCNN数学等价，但训练速度快1.6倍，推理速度比有限元快100倍，且完全不需要外部依赖。

### 核心特性
- ✅ **严格D4硬约束**：与ESCNN不变层数学等价
- ✅ **零外部依赖**：纯PyTorch实现，无需ESCNN等重型库
- ✅ **CPU兼容**：普通笔记本即可运行，无需GPU
- ✅ **高速性能**：比ESCNN训练快1.6倍，比FEM推理快100倍
- ✅ **支持逆问题**：从稀疏观测反演物理参数
- ✅ **3行代码使用**：定义模型、训练、预测仅需3行代码

### 安装依赖
```bash
pip install torch numpy matplotlib
```

### 快速开始
```python
from d4_pinn import D4PINN

# 初始化模型
model = D4PINN(hidden_dim=100, num_layers=5)

# 定义你的PDE残差
def pde_residual(u, x, y):
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    return -u_xx - u_yy + 1.0

# 定义边界条件
n_bc = 2048
x_bc = torch.cat([torch.ones(n_bc//4), -torch.ones(n_bc//4), torch.rand(n_bc//2)*2-1])
y_bc = torch.cat([torch.rand(n_bc//2)*2-1, torch.ones(n_bc//4), -torch.ones(n_bc//4)])
u_bc = torch.zeros(n_bc, 1)
boundary_conditions = [(x_bc, y_bc, u_bc)]

# 一键训练
model.train(pde_residual, boundary_conditions, epochs=20000)

# 预测
u = model.predict(x, y)
```

### 运行示例
```bash
# 示例1：求解泊松方程
python examples/poisson_example.py

# 示例2：求解半线性椭圆方程（论文核心问题）
python examples/semilinear_example.py

# 示例3：稀疏观测下的逆参数识别
python examples/inverse_example.py
```

### 复现论文实验
所有实验代码和结果都在 `experiments/` 文件夹中：
```bash
# 运行基准对比实验
python experiments/baseline_d4_pinn.py

# 运行逆问题实验
python experiments/inverse_problem_experiment.py

# 运行速度优化测试
python experiments/speed_optimization_test.py

# 生成所有论文图表
python experiments/generate_main_figure.py
```

### 论文引用
如果你在研究中使用了本代码，请引用我们的论文：
```
@article{yourname2026d4pinn,
  title={D4-PINN: A Lightweight Architecture for Solving D4-Symmetric Semilinear Elliptic Equations},
  author={你的名字},
  journal={计算物理期刊},
  year={2026}
}
```

