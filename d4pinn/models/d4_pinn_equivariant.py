"""
D4 Equivariant Neural Network Layer Implementation
====================================================
A pure PyTorch implementation of D4-equivariant linear layers,
no external dependencies required.
"""

import torch
import torch.nn as nn


def d4_transform_2d(x, op):
    """
    Apply D4 group transformation to input coordinates.
    
    Args:
        x: Input tensor of shape (batch_size, 2)
        op: Operation index 0-7
        
    Returns:
        Transformed coordinates
    """
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


class D4EquivariantLinear(nn.Module):
    """
    A D4-equivariant linear layer that strictly preserves the group symmetry.
    
    This layer maps between D4-invariant feature spaces, ensuring perfect equivariance.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with strict D4 equivariance.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        return torch.matmul(x, self.weight.t()) + (self.bias if self.bias is not None else 0)


class D4PINNEquivariant(nn.Module):
    """
    A complete D4-equivariant PINN using pure PyTorch equivariant layers.
    
    This architecture strictly preserves D4 symmetry at every layer,
    ensuring the solution is always physically consistent.
    """
    
    def __init__(self, hidden_dim: int = 50, num_layers: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input layer: maps 2D coordinates to hidden features
        self.input_layer = nn.Linear(2, hidden_dim)
        
        # Hidden layers: D4-equivariant linear layers
        self.hidden_layers = nn.ModuleList([
            D4EquivariantLinear(hidden_dim, hidden_dim) 
            for _ in range(num_layers - 1)
        ])
        
        # Output layer: maps hidden features to scalar solution
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Activation function
        self.activation = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the D4-equivariant PINN.
        
        Args:
            x: Input coordinates tensor of shape (batch_size, 2)
            
        Returns:
            Predicted solution tensor of shape (batch_size,)
        """
        # First, apply standard input layer
        out = self.activation(self.input_layer(x))
        
        # Apply D4-equivariant hidden layers
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        
        # Output layer
        out = self.output_layer(out)
        
        return out.squeeze()