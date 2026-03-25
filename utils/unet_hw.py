"""UNet 输入空间尺寸：供 multimodal 与 student_diffusion 共用，避免 model 依赖 pipeline 子模块。"""
import math

from param import STUDENT_SIZE


def unet_sample_hw(net) -> tuple[int, int]:
    """
    送入 UNet 的 (H,W)：优先读 config.sample_size / image_size；缺失时用 param.STUDENT_SIZE。
    再对齐到 8 的倍数，避免奇数尺寸经 downsample 后出现 4 vs 3 等 skip 拼接错误。
    """
    cfg = net.config
    ss = getattr(cfg, 'sample_size', None)
    if ss is None:
        ss = getattr(cfg, 'image_size', None)
    if ss is None:
        th = tw = max(8, int(STUDENT_SIZE))
    elif isinstance(ss, int):
        th = tw = int(ss)
    elif isinstance(ss, (list, tuple)) and len(ss) >= 2:
        th, tw = int(ss[0]), int(ss[1])
    else:
        th = tw = max(8, int(STUDENT_SIZE))

    def _snap8(x: int) -> int:
        x = max(8, x)
        if x % 8 == 0:
            return x
        return int(math.ceil(x / 8.0) * 8)

    return (_snap8(th), _snap8(tw))
