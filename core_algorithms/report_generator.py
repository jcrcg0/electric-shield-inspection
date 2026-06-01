import open3d as o3d
import numpy as np
import os
import base64
import requests

class ReportGenerator:
    def __init__(self, output_dir="data/extracted_objects/views"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # ⚠️ 部署时替换为你真实的大模型 API 密钥和 URL
        self.api_key = "YOUR_VLM_API_KEY" 
        self.api_url = "https://api.example.com/v1/chat/completions" 

    def capture_multi_views(self, pcd, device_name):
        print(f"📷 正在为 {device_name} 自动拍摄多视角分析图...")
        image_paths = []
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=800, height=800)
        vis.add_geometry(pcd)
        ctr = vis.get_view_control()
        
        for i in range(4):
            ctr.rotate(900.0, 0.0)
            vis.poll_events()
            vis.update_renderer()
            img_path = os.path.join(self.output_dir, f"{device_name}_view_{i}.png")
            vis.capture_screen_image(img_path)
            image_paths.append(img_path)
            
        vis.destroy_window()
        return image_paths

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_report(self, image_paths, device_name):
        print("🧠 正在请求多模态大模型分析异常...")
        if self.api_key == "YOUR_VLM_API_KEY":
            return "[系统提示] 未配置大模型 API Key。系统已高亮红色异常区域，形态疑似为表面缺损或外部异物搭挂，请人工结合3D视图进行复核。"

        system_prompt = (
            f"你是电力设备巡检AI专家。这是 {device_name} 3D点云的多视角截图。"
            "灰色是正常结构，**红色是算法定位的异常（如碰撞缺损、鸟巢等异物）**。"
            "请观察红色区域给出专业的简短诊断报告。"
        )

        content = [{"type": "text", "text": system_prompt}]
        for path in image_paths:
            base64_img = self._encode_image(path)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}})

        payload = {
            "model": "qwen-vl-plus", 
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 500
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"❌ 调用大模型失败: {e}"