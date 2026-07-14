from pathlib import Path

import cv2
import numpy as np
from warnings import warn
from constants.enum_keys import PG
from pgdataset.s1_skeleton import PgdSkeleton
import pred.gesture_pred


# ---------------------------------------------------------------------------
# 替代 imgaug 的工具函数（避免安装 imgaug 依赖）
# ---------------------------------------------------------------------------

def _draw_keypoints_on_image(img, coords, color=(0, 255, 0), radius=3):
    """在图像上绘制关键点（替代 imgaug.KeypointsOnImage.draw_on_image）。

    Args:
        img: numpy 图像 (H, W, 3)
        coords: 形状 (J, 2)，每行为 (x, y) 像素坐标
        color: BGR 颜色元组
        radius: 关键点半径
    """
    out = img.copy()
    for x, y in coords:
        if x > 0 and y > 0:
            cv2.circle(out, (int(x), int(y)), radius, color, -1)
    return out


def _draw_text_on_image(img, y, x, text, color=(255, 50, 50), size=1.0):
    """在图像上绘制文字（替代 imgaug.imgaug.draw_text）。

    Args:
        img: numpy 图像
        y, x: 文字左上角像素坐标
    """
    out = img.copy()
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)
    return out


# ---------------------------------------------------------------------------
# 替代 aichallenger.s1_resize.ResizeKeepRatio
# ---------------------------------------------------------------------------

class _ResizeKeepRatio:
    """保持宽高比的缩放器（替代 aichallenger.s1_resize.ResizeKeepRatio）。"""

    def __init__(self, target_size):
        self.target_size = target_size

    def resize(self, img, *_args, **_kwargs):
        """缩放并居中放置，返回 (resized_img, unused, unused)。"""
        h, w = img.shape[:2]
        tw, th = self.target_size
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.zeros((th, tw, 3), dtype=img.dtype)
        dx = (tw - nw) // 2
        dy = (th - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        return canvas, None, None


class Player:
    def __init__(self, is_unittest=False):
        self.img_size = (512, 512)
        self.gpred = pred.gesture_pred.GesturePred()
        self.is_unittest = is_unittest

    def play_dataset_video(self, is_train, video_index, show=True):
        self.scd = PgdSkeleton(Path.home() / 'PoliceGestureLong', is_train, self.img_size)
        res = self.scd[video_index]
        print('Playing %s' % res[PG.VIDEO_NAME])
        coord_norm_FXJ = res[PG.COORD_NORM]  # Shape: F,X,J
        coord_norm_FJX = np.transpose(coord_norm_FXJ, (0, 2, 1))  # FJX
        coord = coord_norm_FJX * np.array(self.img_size)  # (frames, J, 2)
        cap = cv2.VideoCapture(str(res[PG.VIDEO_PATH]))
        v_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = int(1000/(v_fps*4))
        gestures = []  # Full video gesture recognition results
        for n in range(v_size):
            gdict = self.gpred.from_skeleton(coord_norm_FXJ[n][np.newaxis])
            gesture = gdict[PG.OUT_ARGMAX]
            gestures.append(gesture)
            if not show:
                continue
            ret, img = cap.read()
            re_img = cv2.resize(img, self.img_size)
            ges_name = self.gesture_dict[gesture]
            re_img = _draw_text_on_image(re_img, 50, 100, ges_name, (255, 50, 50), size=1.0)
            img_kps = _draw_keypoints_on_image(re_img, coord[n], color=(0, 255, 0))
            if self.is_unittest:
                break
            cv2.imshow("Play saved keypoint results", img_kps)
            cv2.waitKey(duration)
        cap.release()
        gestures = np.array(gestures, int)
        res[PG.PRED_GESTURES] = gestures
        print('The prediction of video ', res[PG.VIDEO_NAME], ' is completed')
        return res

    def play_custom_video(self, video_path):
        """video_path string: play video on disk
            video_path None: play video from camera on realtime
        """
        rkr = _ResizeKeepRatio((512, 512))
        if video_path is None:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise IOError('Failed to open camera.')
        else:
            cap = cv2.VideoCapture(str(video_path))
            v_fps = int(cap.get(cv2.CAP_PROP_FPS))
            if v_fps != 15:
                warn('Suggested video frame rate is 15, currently %d, which may impact accuracy' % v_fps)
        duration = 10
        while True:
            ret, img = cap.read()
            if not ret:
                break
            re_img, _, _ = rkr.resize(img, np.zeros((2,)), np.zeros((4,)))
            # re_img = cv2.resize(img, self.img_size)
            gdict = self.gpred.from_img(re_img)
            gesture = gdict[PG.OUT_ARGMAX]
            # Keypoints on image
            coord_norm_FXJ = gdict[PG.COORD_NORM]
            coord_norm_FJX = np.transpose(coord_norm_FXJ, (0, 2, 1))  # FJX
            coord_FJX = coord_norm_FJX * np.array(self.img_size)
            re_img = _draw_keypoints_on_image(re_img, coord_FJX[0], color=(0, 255, 0))
            # Gesture name on image
            ges_name = self.gesture_dict[gesture]
            re_img = _draw_text_on_image(re_img, 50, 100, ges_name, (255, 50, 50), size=1.0)
            if self.is_unittest:
                break
            cv2.imshow("Play saved keypoint results", re_img)
            cv2.waitKey(duration)
        cap.release()


    gesture_dict = {
        0: "NO GESTURE",
        1: "STOP",
        2: "MOVE STRAIGHT",
        3: "LEFT TURN",
        4: "LEFT TURN WAITING",
        5: "RIGHT TURN",
        6: "LANG CHANGING",
        7: "SLOW DOWN",
        8: "PULL OVER"}

    gesture_dict_c = {
        0: "无手势",
        1: "停止",
        2: "直行",
        3: "左转",
        4: "左待转",
        5: "右转",
        6: "变道",
        7: "减速",
        8: "靠边停车"}