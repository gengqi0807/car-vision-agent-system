"""
测试 CTPGREngine：用随机生成的假图片进行单帧推理，验证模型加载和推理链路是否正常。
"""
import sys
import numpy as np

from ctpgr_engine import CTPGREngine


def main():
    print("[enginetest] 开始初始化 CTPGREngine，加载姿态估计 + 手势 RNN 模型...")
    engine = CTPGREngine()
    print("[enginetest] 模型加载完成。")

    # 生成一张 256x256x3 的随机假图片（BGR, uint8）
    fake_frame = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)

    print("[enginetest] 正在进行假图片推理...")
    result = engine.predict_frame(fake_frame)

    print("[enginetest] 推理完成！")
    print(f"  → 手势: {result['gesture']}")
    print(f"  → 置信度: {result['confidence']:.4f}")
    print(f"  → 关键点数量: {len(result['keypoints'])}")
    print()

    # 测试 reset_state 是否正常
    print("[enginetest] 测试 reset_state()...")
    engine.reset_state()
    result2 = engine.predict_frame(fake_frame)
    print(f"  → 重置后手势: {result2['gesture']}")
    print(f"  → 重置后置信度: {result2['confidence']:.4f}")

    print("\n[enginetest] 全部测试通过！")


if __name__ == "__main__":
    main()
