import os
import openxlab
openxlab.login(ak='xvapgvld3wl5dgnv4dyg', sk='w6l9dnlrmdwkn5pvl6m2eok9a70gpaezmbzy4ybx')

from openxlab.dataset import get

DATASET = "OpenDataLab/NVGesture"
LOCAL_ROOT = r"D:\datasets\NVGesture"     # ← 改成真实目录
os.makedirs(LOCAL_ROOT, exist_ok=True)

# 1) 重新下载（无 force 参数；SDK 自动 SHA256 校验并补全缺失/损坏文件）
get(dataset_repo=DATASET, target_path=LOCAL_ROOT)

# 2) 核对：绕过有 bug 的 query，直接调底层 API 枚举【全部】文件
from openxlab.dataset.commands.utility import ContextInfoNoLogin
ctx = ContextInfoNoLogin()
client = ctx.get_client()
parsed = DATASET.replace("/", ",")         # "OpenDataLab,NVGesture"

all_files, after, has_more = [], None, True
while has_more:
    d = client.get_api().get_dataset_files(
        dataset_name=parsed, payload={}, after=after, limit=1000
    )
    all_files.extend(d['list'])
    has_more = d.get('hasNext', False)
    after = d.get('after') if has_more else None

# 3) 比对本地：SDK 会在 target_path 下再建一层 OpenDataLab___NVGesture
SAVE_SUB = DATASET.replace("/", "___")
missing = [
    f['path']
    for f in all_files
    if not os.path.exists(os.path.join(LOCAL_ROOT, SAVE_SUB, f['path'].lstrip('/')))
]

print("远程文件总数:", len(all_files))
print("本地缺失文件数:", len(missing))
for m in missing:
    print("缺失:", m)


