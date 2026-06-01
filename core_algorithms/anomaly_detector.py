import torch
import numpy as np
import open3d as o3d
import os
from models.Point_MAE import Point_MAE

# --- 官方模型所需的配置字典 ---
class TransformerConfig:
    mask_ratio = 0.0 # 推理时全可见，不遮蔽
    trans_dim = 384
    depth = 12
    drop_path_rate = 0.1
    num_heads = 6
    encoder_dims = 256
    mask_type = 'rand'
    decoder_depth = 4
    decoder_num_heads = 6

class PointMAEConfig:
    transformer_config = TransformerConfig()
    group_size = 32
    num_group = 64
    loss = 'cdl2'

class AnomalyDetector:
    def __init__(self, weight_path, num_points=2048, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.num_points = num_points
        self.device = device
        print(f"🔧 初始化真实 Point-MAE 模型 (设备: {self.device})...")
        
        # 1. 实例化真实的模型
        config = PointMAEConfig()
        self.model = Point_MAE(config).to(self.device)
        
        # 2. 尝试加载真实权重
        if os.path.exists(weight_path):
            checkpoint = torch.load(weight_path, map_location=self.device)
            # 兼容官方保存格式，剔除 module. 前缀
            if 'model_state_dict' in checkpoint:
                base_ckpt = checkpoint['model_state_dict']
            else:
                base_ckpt = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            self.model.load_state_dict(base_ckpt, strict=False)
            print(f"✅ 专属设备权重加载完成: {weight_path}")
        else:
            print(f"⚠️ 警告: 未找到权重 {weight_path}，当前使用随机初始化权重（请先去训练中心训练）。")
            
        self.model.eval()

    def farthest_point_sample(self, xyz, npoint):
        idx = np.random.choice(xyz.shape[0], npoint, replace=xyz.shape[0] < npoint)
        return xyz[idx]

    def detect(self, pcd):
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return pcd, np.array([])

        # 1. 预处理与归一化
        points_sampled = self.farthest_point_sample(points, self.num_points)
        centroid = np.mean(points_sampled, axis=0)
        points_centered = points_sampled - centroid
        max_distance = np.max(np.sqrt(np.sum(points_centered**2, axis=1)))
        if max_distance == 0: max_distance = 1e-6
        points_normalized = points_centered / max_distance

        pt_tensor = torch.tensor(points_normalized, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 2. 真实模型前向推理
        with torch.no_grad():
            neighborhood, center = self.model.group_divider(pt_tensor)
            x_vis, mask = self.model.MAE_encoder(neighborhood, center, noaug=True) 
            B, _, C = x_vis.shape
            
            pos_emd_vis = self.model.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            x_full = x_vis 
            pos_full = pos_emd_vis
            
            x_rec = self.model.MAE_decoder(x_full, pos_full, return_token_num=x_full.shape[1])
            B, M, C = x_rec.shape
            rebuild_points = self.model.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)
            reconstructed_points = rebuild_points.squeeze(0).cpu().numpy()

        # 3. 计算误差并标红
        pcd_input = o3d.geometry.PointCloud()
        pcd_input.points = o3d.utility.Vector3dVector(points_normalized)
        pcd_recon = o3d.geometry.PointCloud()
        pcd_recon.points = o3d.utility.Vector3dVector(reconstructed_points)
        
        distances = np.asarray(pcd_input.compute_point_cloud_distance(pcd_recon))
        threshold = np.mean(distances) + 2 * np.std(distances)
        anomaly_mask = distances > threshold

        colors = np.zeros((self.num_points, 3))
        colors[:] = [0.7, 0.7, 0.7] 
        colors[anomaly_mask] = [1.0, 0.0, 0.0] 
        pcd_input.colors = o3d.utility.Vector3dVector(colors)
        
        points_restored = (np.asarray(pcd_input.points) * max_distance) + centroid
        pcd_input.points = o3d.utility.Vector3dVector(points_restored)

        print(f"🚨 真实模型检测完成！共发现 {np.sum(anomaly_mask)} 个异常点。")
        return pcd_input, anomaly_mask