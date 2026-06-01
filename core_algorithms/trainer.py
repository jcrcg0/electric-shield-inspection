import os
import open3d as o3d
import numpy as np
import torch
import gradio as gr
import time
from models.Point_MAE import Point_MAE
# 复用检测器中的配置
from core_algorithms.anomaly_detector import PointMAEConfig 

class PointMAETrainer:
    def __init__(self, weights_dir="models/weights"):
        self.weights_dir = weights_dir
        os.makedirs(self.weights_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_points = 2048

    def farthest_point_sample(self, xyz, npoint):
        idx = np.random.choice(xyz.shape[0], npoint, replace=xyz.shape[0] < npoint)
        return xyz[idx]

    def prepare_training_data(self, std_pcd):
        """将 Open3D 点云转换为 PyTorch 训练所需的 Tensor 格式"""
        points = np.asarray(std_pcd.points)
        points_sampled = self.farthest_point_sample(points, self.num_points)
        
        # 归一化
        centroid = np.mean(points_sampled, axis=0)
        points_centered = points_sampled - centroid
        max_distance = np.max(np.sqrt(np.sum(points_centered**2, axis=1)))
        if max_distance == 0: max_distance = 1e-6
        points_normalized = points_centered / max_distance
        
        return torch.tensor(points_normalized, dtype=torch.float32).unsqueeze(0).to(self.device)

    def train_model(self, device_name, std_model_path, epochs, progress=gr.Progress()):
        if not os.path.exists(std_model_path):
            yield f"❌ 错误：找不到标准模型 {std_model_path}", None
            return

        yield "⏳ 正在加载标准模型并初始化真实的 PyTorch 训练引擎...", None
        
        # 1. 准备数据
        std_pcd = o3d.io.read_point_cloud(std_model_path)
        pt_tensor = self.prepare_training_data(std_pcd) # [1, 2048, 3]

        # 2. 初始化真实模型与优化器
        config = PointMAEConfig()
        # 训练时开启高遮蔽率 (迫使模型学习)
        config.transformer_config.mask_ratio = 0.6 
        model = Point_MAE(config).to(self.device)
        model.train()
        
        # 使用主流的 AdamW 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)

        save_path = os.path.join(self.weights_dir, f"{device_name}_point_mae.pth")
        
        # 3. 真实的训练循环 (反向传播)
        for epoch in progress.tqdm(range(int(epochs)), desc=f"深度学习训练中: {device_name}"):
            optimizer.zero_grad()
            
            # 由于我们没有批量数据集，此处用小微扰动(抖动)实现实时数据增强
            jittered_tensor = pt_tensor + torch.randn_like(pt_tensor) * 0.01
            
            # 前向传播并计算 Chamfer Loss
            loss = model(jittered_tensor)
            
            # 反向传播与权重更新
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                log_msg = f"🔄 Epoch [{epoch}/{epochs}] - 真实 Chamfer 距离 Loss: {loss.item():.4f}"
                yield log_msg, None

        # 4. 保存真实的模型权重字典 (.pth)
        torch.save(model.state_dict(), save_path)

        success_msg = f"✅ 训练圆满完成！(Loss 降至: {loss.item():.4f})\n专属权重已保存至: {save_path}\n您现在可以前往【智能巡检大厅】检测带病设备了。"
        yield success_msg, save_path