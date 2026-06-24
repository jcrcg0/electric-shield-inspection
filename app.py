import gradio as gr
import yaml
import os
import numpy as np
import open3d as o3d
import torch
import shutil

# 导入底层算法模块
from utils.point_cloud_tools import load_point_cloud, save_point_cloud
from core_algorithms.extractor import extract_equipment
from core_algorithms.anomaly_detector import AnomalyDetector
from core_algorithms.report_generator import ReportGenerator
from core_algorithms.trainer import PointMAETrainer

CONFIG_PATH = "configs/config.yaml"

# ==========================================
# 工具函数：动态配置管理 (零代码配置的核心)
# ==========================================
def load_config():
    if not os.path.exists(CONFIG_PATH):
        # 如果配置文件丢失，自动生成基础模板
        os.makedirs("configs", exist_ok=True)
        base_config = {"project_root": os.getcwd(), "equipments": {}}
        save_config(base_config)
        return base_config
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config_data):
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config_data, f, allow_unicode=True, sort_keys=False)

def get_device_choices():
    config = load_config()
    return [v['name'] for k, v in config.get('equipments', {}).items()]

# ==========================================
# 核心管线 1：智能检测流水线
# ==========================================
def process_pipeline(input_file_path, target_device_name):
    if input_file_path is None:
        return None, None, [], "❌ 请先上传点云文件。"
    
    config = load_config()
    project_root = config.get('project_root', os.getcwd())
    
    # 根据中文名反查设备 ID
    target_key = next((k for k, v in config['equipments'].items() if v['name'] == target_device_name), None)
    if not target_key:
        return None, None, [], "❌ 系统配置库中未找到该设备，请先去训练中心注册。"
        
    device_cfg = config['equipments'][target_key]
    
    try:
        # 1. 提取与分割
        raw_pcd = load_point_cloud(input_file_path)
        std_model_path = os.path.join(project_root, device_cfg['standard_model_path'])
        
        if not os.path.exists(std_model_path):
            return None, None, [], f"❌ 找不到该设备的标准模型: {std_model_path}"
            
        std_pcd = load_point_cloud(std_model_path)
        extracted_pcd, _ = extract_equipment(raw_pcd, std_pcd, device_cfg.get('voxel_size', 0.05))
        
        os.makedirs(os.path.join(project_root, "data/extracted_objects"), exist_ok=True)
        ext_path = os.path.join(project_root, "data/extracted_objects", f"{target_key}_extracted.ply")
        save_point_cloud(extracted_pcd, ext_path)
        
        # 2. 异常检测
        detector = AnomalyDetector(weight_path=device_cfg['model_weight_path'])
        analyzed_pcd, anomaly_mask = detector.detect(extracted_pcd)
        
        res_path = os.path.join(project_root, "data/extracted_objects", f"{target_key}_result.ply")
        save_point_cloud(analyzed_pcd, res_path)
        
        # 3. 多视角抓拍与大模型报告
        reporter = ReportGenerator()
        view_images = reporter.capture_multi_views(analyzed_pcd, device_cfg['name'])
        
        if np.sum(anomaly_mask) > 0:
            final_report = reporter.generate_report(view_images, device_cfg['name'])
        else:
            final_report = f"✅ 【检测结果】: {device_cfg['name']} 状态良好，表面未见明显异常。"

        return ext_path, res_path, view_images, final_report

    except Exception as e:
        return None, None, [], f"❌ 处理发生错误: {str(e)}"

# ==========================================
# 核心管线 2：自助训练与设备动态注册
# ==========================================
def train_and_register_pipeline(device_id, device_name, std_file, epochs, progress=gr.Progress()):
    if not device_id or not device_name or not std_file:
        yield "❌ 请填写完整的设备代号、名称，并上传标准模型。", None, gr.update()
        return
        
    device_id = device_id.strip().lower()
    config = load_config()
    project_root = config.get('project_root', os.getcwd())
    
    # 1. 保存用户上传的标准模型
    std_dir = os.path.join(project_root, "data/standard_models")
    os.makedirs(std_dir, exist_ok=True)
    std_save_path = os.path.join(std_dir, f"{device_id}_std.ply")
    shutil.copy(std_file.name, std_save_path)
    
    # 2. 调用底层大模型训练器
    trainer = PointMAETrainer()
    weight_save_path = ""
    for log_msg, weight_path in trainer.train_model(device_id, std_save_path, epochs, progress):
        weight_save_path = weight_path if weight_path else weight_save_path
        yield log_msg, None, gr.update()
    
    # 3. ⭐️ 核心：零代码动态注册设备到 config.yaml
    config['equipments'][device_id] = {
        'name': device_name,
        'extraction_method': "registration",
        'standard_model_path': f"data/standard_models/{device_id}_std.ply",
        'model_weight_path': f"models/weights/{device_id}_point_mae.pth",
        'voxel_size': 0.05
    }
    save_config(config)
    
    # 4. 刷新前端的下拉菜单
    updated_choices = get_device_choices()
    success_msg = f"✅ 训练与注册成功！\n设备【{device_name}】已自动添加至系统库，请前往巡检大厅使用。"
    
    yield success_msg, weight_save_path, gr.update(choices=updated_choices, value=device_name)


# ==========================================
# UI 界面构建
# ==========================================
def create_ui():
    initial_choices = get_device_choices()
    if not initial_choices:
        initial_choices = ["请先前往训练中心添加设备"]

    with gr.Blocks(title="电力点云 AI 系统", theme=gr.themes.Base()) as demo:
        gr.Markdown("# ⚡ 零代码电力点云缺损异物检测系统")
        
        with gr.Tabs():
            # ==============================
            # Tab 1: 智能巡检大厅
            # ==============================
            with gr.Tab("🔍 智能巡检大厅"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📂 任务下发")
                        input_file = gr.File(label="上传现场杂乱点云 (.ply, .las)")
                        # 注意这里的 dropdown 我们赋予了一个变量名，方便后续被其他 Tab 动态更新
                        target_dropdown = gr.Dropdown(choices=initial_choices, value=initial_choices[0], label="选择检测目标", interactive=True)
                        run_btn = gr.Button("🚀 启动全自动智能巡检", variant="primary")
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### 📑 AI 多模态诊断报告")
                        report_output = gr.Markdown("等待执行...")
                        gallery_output = gr.Gallery(label="系统高亮抓拍图", columns=4, height=200)
                        
                gr.Markdown("---")
                gr.Markdown("### 🧊 3D 点云空间分析")
                with gr.Row():
                    with gr.Column():
                        model_extracted = gr.Model3D(clear_color=[0, 0, 0, 0], label="[阶段1] 自动物理裁剪结果")
                    with gr.Column():
                        model_result = gr.Model3D(clear_color=[0, 0, 0, 0], label="[阶段2] Point-MAE 异常重建热力图")
                
                run_btn.click(fn=process_pipeline, inputs=[input_file, target_dropdown], 
                              outputs=[model_extracted, model_result, gallery_output, report_output])

            # ==============================
            # Tab 2: 零代码设备训练中心
            # ==============================
            with gr.Tab("⚙️ 自助模型训练中心 (自动注册)"):
                gr.Markdown("💡 **说明**：此页面专为业务人员设计。只需上传一个完美无缺的新设备点云，系统将自动进行深度学习训练，并**自动将其注册到检测菜单中**，全程无需修改任何代码。")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 设备建档")
                        new_dev_id = gr.Textbox(label="设备英文代号 (作为系统内部ID，如: insulator_v2)", placeholder="限英文字母和下划线")
                        new_dev_name = gr.Textbox(label="设备中文名称 (展示给用户，如: 新型绝缘子串)", placeholder="用于界面展示")
                        std_model_upload = gr.File(label="上传标准纯净点云模型 (.ply)")
                        epoch_slider = gr.Slider(minimum=10, maximum=300, step=10, value=50, label="训练强度 (Epochs)")
                        train_btn = gr.Button("🔨 开始训练并注册入库", variant="secondary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 📈 系统运行终端")
                        train_log = gr.Textbox(label="实时状态与损失计算", lines=12, interactive=False)
                        weight_output = gr.File(label="训练产物 (自动挂载至后台)")
                        
                # 绑定事件：注意这里的 output 包含了 target_dropdown，用于实时刷新巡检大厅的菜单
                train_btn.click(
                    fn=train_and_register_pipeline,
                    inputs=[new_dev_id, new_dev_name, std_model_upload, epoch_slider],
                    outputs=[train_log, weight_output, target_dropdown]
                )

            # ==============================
            # Tab 3: 系统数据库查看 (纯展示)
            # ==============================
            with gr.Tab("📊 设备数据库库房"):
                gr.Markdown("💡 这里展示当前系统已经支持和训练好的所有设备类型。")
                
                def load_db_view():
                    cfg = load_config()
                    equipments = cfg.get('equipments', {})
                    if not equipments: return "当前数据库为空。"
                    db_str = ""
                    for k, v in equipments.items():
                        db_str += f"**{v['name']} (ID: {k})**\n"
                        db_str += f"- 标准模型路径: `{v['standard_model_path']}`\n"
                        db_str += f"- 权重文件路径: `{v['model_weight_path']}`\n\n"
                    return db_str
                    
                db_display = gr.Markdown(load_db_view())
                refresh_btn = gr.Button("🔄 刷新数据库视图")
                refresh_btn.click(fn=load_db_view, outputs=[db_display])

    return demo

if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)