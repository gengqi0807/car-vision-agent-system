import torch
import cv2
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from models.pose_estimation_model import PoseEstimationModel

# 加载模型
print("加载姿态模型...")
pose_model = PoseEstimationModel()
pose_path = os.path.join(BASE_DIR, 'weigth', 'pose_model.pt')
pose_ckpt = torch.load(pose_path, map_location='cpu')
pose_model.load_state_dict(pose_ckpt, strict=False)
pose_model.eval()
pose_model.to('cpu')
print("模型加载完成")

# 生成一张随机测试图片
test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
norm_img = test_img.astype(np.float32) / 255.0
norm_img = np.transpose(norm_img, axes=(2, 0, 1))[np.newaxis]
norm_img = torch.from_numpy(norm_img).float()

# 推理
with torch.no_grad():
    res = pose_model(norm_img)

# 打印输出字典的所有键
print("模型输出的键:", res.keys())