"""
police — 交警手势识别模块

模块结构：
  config             : 所有可调参数集中管理
  geometry           : 几何计算、坐标系、区域划分、摆动检测
  models             : MediaPipe Pose/Hand 模型加载与推理
  features           : 逐帧特征提取
  gesture_classifier : 状态机 + 8 种手势分类决策树
  visualization      : 骨架绘制、中文文字叠加
  main               : 主循环整合
"""

__version__ = "5.0.0"
