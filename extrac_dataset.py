import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_array, coo_matrix
import random
import matplotlib
import os

matplotlib.use('TkAgg')


def visualize_hsi_lidar_data(hsi, lidar, train, test, rgb=None):
    """
    可视化高光谱、LiDAR数据以及训练测试标签

    参数:
    - hsi: numpy array, shape (N, H, W) - 高光谱数据
    - lidar: numpy array, shape (C, H, W) - LiDAR数据
    - train: scipy.sparse.coo_array, shape (H, W) - 训练标签
    - test: scipy.sparse.coo_array, shape (H, W) - 测试标签
    - rgb: numpy array, shape (3, H, W) - 可选，RGB 数据（CHW）
    """
    # 将稀疏矩阵转换为密集矩阵用于可视化
    train_dense = train.toarray()
    test_dense = test.toarray()

    # 创建子图：增加一个 RGB 视图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Hyperspectral & LiDAR Data Visualization', fontsize=16)

    rgb_bands = [5, 3, 1]  # 转换为0索引
    H, W = hsi.shape[1], hsi.shape[2]

    # 提取RGB波段并归一化
    r_band = hsi[rgb_bands[0]]
    g_band = hsi[rgb_bands[1]]
    b_band = hsi[rgb_bands[2]]

    # 归一化到[0,1]范围
    r_norm = (r_band - r_band.min()) / (r_band.max() - r_band.min() + 1e-8)
    g_norm = (g_band - g_band.min()) / (g_band.max() - g_band.min() + 1e-8)
    b_norm = (b_band - b_band.min()) / (b_band.max() - b_band.min() + 1e-8)

    # 合成RGB图像
    rgb_image = np.stack([r_norm, g_norm, b_norm], axis=-1)

    # 显示RGB合成图
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title(f'HSI RGB Composite ({rgb_bands[0]}, {rgb_bands[1]}, {rgb_bands[2]})')

    # 显示LiDAR数据（显示第一个通道，灰度图）
    im2 = axes[0, 1].imshow(lidar[0], cmap='gray')
    axes[0, 1].set_title('LiDAR Data')
    plt.colorbar(im2, ax=axes[0, 1])

    # 可选：显示真实 RGB（CHW）
    if rgb is not None:
        rgb_arr = _to_numpy(rgb)
        rgb_chw = np.array(rgb_arr)
        if rgb_chw.ndim != 3 or rgb_chw.shape[0] != 3:
            raise ValueError(f"rgb 期望为 (3,H,W)，实际 shape={rgb_chw.shape}")
        # (3,H,W) -> (H,W,3)
        rgb_hwc = rgb_chw.transpose(1, 2, 0)
        r = rgb_hwc[..., 0]
        g = rgb_hwc[..., 1]
        b = rgb_hwc[..., 2]
        r_norm = (r - r.min()) / (r.max() - r.min() + 1e-8)
        g_norm = (g - g.min()) / (g.max() - g.min() + 1e-8)
        b_norm = (b - b.min()) / (b.max() - b.min() + 1e-8)
        rgb_vis = np.stack([r_norm, g_norm, b_norm], axis=-1)
        axes[0, 2].imshow(rgb_vis)
        axes[0, 2].set_title('RGB (field)')
    else:
        axes[0, 2].axis('off')

    # 显示训练标签
    im3 = axes[1, 0].imshow(train_dense, cmap='tab20', vmin=0, vmax=max(1, train_dense.max()))
    axes[1, 0].set_title('Training Labels')
    plt.colorbar(im3, ax=axes[1, 0])

    # 显示测试标签
    im4 = axes[1, 1].imshow(test_dense, cmap='tab20', vmin=0, vmax=max(1, test_dense.max()))
    axes[1, 1].set_title('Test Labels')
    plt.colorbar(im4, ax=axes[1, 1])

    axes[1, 2].axis('off')

    # 调整子图间距
    plt.tight_layout()
    plt.show()


def check_invalid_bands(hsi_data):
    """
    检查高光谱数据立方体中是否存在无效波段。
    hsi_data: numpy array, shape (H, W, D)
    """
    if hsi_data.ndim != 3:
        raise ValueError("输入数据必须是三维数组 (H, W, D)")

    H, W, D = hsi_data.shape
    invalid_bands = []

    print(f"数据形状: {hsi_data.shape}")
    print(f"数据类型: {hsi_data.dtype}")

    for i in range(D):
        band = hsi_data[:, :, i]

        # 1. 检查是否全为 NaN 或 0
        if np.all(np.isnan(band)) or np.all(band == 0):
            invalid_bands.append((i, "全零或全NaN"))
            continue

        # 2. 检查是否全为同一个常数 (方差为0)
        if np.var(band) == 0:
            invalid_bands.append((i, "常数波段 (方差为0)"))
            continue

        # 3. (可选) 检查信噪比极低的情况
        # 如果均值很小但标准差也很小，可能是噪声或填充
        mean_val = np.mean(band)
        std_val = np.std(band)
        if mean_val == 0 and std_val == 0:
            invalid_bands.append((i, "均值为0且标准差为0"))

    if invalid_bands:
        print("\n发现潜在的非光谱/无效波段索引:")
        for idx, reason in invalid_bands:
            print(f"  波段 {idx}: {reason}")
        return [idx for idx, _ in invalid_bands]
    else:
        print("\n所有波段均包含变化的数值，看起来都是有效的光谱数据。")
        return []


def get_coo_subset(coo_obj, n_samples):
    """
    从 COO 格式的稀疏数组/矩阵中提取前 n_samples 行（样本）。

    参数:
        coo_obj (scipy.sparse.coo_array | scipy.sparse.coo_matrix): 输入的 COO 格式对象。
        n_samples (int): 需要提取的前 N 个样本数量（行数）。

    返回:
        scipy.sparse.coo_array: 包含前 N 行数据的新 COO 数组。
    """

    # 2. 边界检查
    total_rows = coo_obj.shape[0]
    if n_samples <= 0:
        raise ValueError("n_samples 必须大于 0")

    # 如果请求的行数超过总行数，则返回整个数据集的副本（或根据需求截断）
    effective_n = min(n_samples, total_rows)

    # 3. 创建掩码 (Mask)
    # 找出所有行索引小于 effective_n 的非零元素
    # 注意：COO 格式中的数据不一定按行排序，所以不能直接切片 data[:k]
    mask = coo_obj.row < effective_n

    # 4. 提取对应的数据
    new_data = coo_obj.data[mask]
    new_row = coo_obj.row[mask]
    new_col = coo_obj.col[mask]

    # 5. 构建新的 COO 对象
    # 形状设置为 (effective_n, 原始列数)
    # 即使某些行在有效范围内但没有非零元素，形状也应正确反映行数
    new_shape = (effective_n, coo_obj.shape[1])

    subset = coo_array((new_data, (new_row, new_col)), shape=new_shape)

    return subset


def _to_numpy(x):
    """将可能的 torch.Tensor / numpy.ndarray 统一转成 numpy.ndarray。"""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


def _ensure_chw_rgb(rgb_arr):
    """
    将 RGB 数据转为 CHW 格式（通道优先）。
    支持输入形状: (3,H,W) / (H,W,3)。
    """
    import numpy as _np

    rgb = _to_numpy(rgb_arr)
    if not isinstance(rgb, _np.ndarray):
        rgb = _np.array(rgb)

    if rgb.ndim != 3:
        raise ValueError(f"RGB 期望为 3D 数组，实际 ndim={rgb.ndim}, shape={getattr(rgb,'shape',None)}")

    if rgb.shape[0] == 3:
        # (3,H,W)
        return rgb
    if rgb.shape[-1] == 3:
        # (H,W,3) -> (3,H,W)
        return rgb.transpose(2, 0, 1)

    raise ValueError(f"无法识别 RGB 的通道维: shape={rgb.shape}，期望 (3,H,W) 或 (H,W,3)")


def _maybe_extract_rgb_from_extras(extras):
    """从 extras（可能是 dict）里尽量找到 rgb 字段。"""
    if extras is None:
        return None
    if isinstance(extras, dict):
        for key in ["rgb", "RGB", "rgb_hr", "rgb_tensor", "rgb_data", "rgb_patch"]:
            if key in extras:
                return extras[key]
    return None


from rs_fusion_datasets import fetch_houston2018_ouc, Houston2018Ouc

# For fetch_houston2018_ouc, fetch_augsberg_ouc, fetch_berlin_ouc
a = fetch_houston2018_ouc()
# dataset = Houston2018Ouc('train', patch_size=11)
# x_h, x_l, y, extras = dataset[0]
hsi = a[0]
from rs_fusion_datasets.util.transforms import ChannelPCA

# pca = ChannelPCA(n_components=32)
# hsi_pca = pca(hsi)    # 得到 shape=[16, H, W]
# invalid_indices = check_invalid_bands(hsi)
# print(invalid_indices)
lidar = a[1]
train = a[2]
test = a[3]

# RGB 来自拼接结果：G:/Houston2018/Houston2018 raw/RGB_mat 下四个坐标分块横向拼成完整 RGB
rgb_mat_dir = r"G:\Houston2018\Houston2018 raw\RGB_mat"
import glob
import re
import scipy.io as sio

mat_paths = glob.glob(os.path.join(rgb_mat_dir, "*.mat"))
coord_pat = re.compile(r"_(\-?\d+)_(-?\d+)\.mat$")

def _get_x_from_name(p: str) -> int:
    bn = os.path.basename(p)
    m = coord_pat.search(bn)
    if not m:
        raise RuntimeError(f"无法从文件名解析坐标: {bn}")
    return int(m.group(1))

mat_paths_sorted = sorted(mat_paths, key=_get_x_from_name)  # 从左到右按 x 排序

rgb_parts = []
for p in mat_paths_sorted:
    mat_dict = sio.loadmat(p)
    arr = mat_dict["data"]
    rgb_parts.append(arr)

# 拼接：横向（宽度 W）拼 4 个 patch
# arr shape: (H, W, C) -> concat axis=1 -> (H, W*4, C)
rgb_hwc = np.concatenate(rgb_parts, axis=1)

# 转成 CHW： (H, W, C) -> (C, H, W)
rgb_chw = rgb_hwc.transpose(2, 0, 1)

# 二次确认：rgb 的 H/W 是否为其它模态的 10 倍（这里用 hsi_pca 的 H/W 作为参照）
ref_h, ref_w = int(hsi_pca.shape[-2]), int(hsi_pca.shape[-1])
rgb_h, rgb_w = int(rgb_chw.shape[-2]), int(rgb_chw.shape[-1])
ratio_h = rgb_h / max(ref_h, 1)
ratio_w = rgb_w / max(ref_w, 1)
print(
    f"[shape-check] ref(hsi_pca) H,W=({ref_h},{ref_w}) -> rgb H,W=({rgb_h},{rgb_w}) "
    f"ratio=(H:{ratio_h:.3f}, W:{ratio_w:.3f})"
)
if abs(ratio_h - 10.0) > 1e-3 or abs(ratio_w - 10.0) > 1e-3:
    print("[shape-check][WARN] rgb 的 H/W 未严格等于其它模态的 10 倍。")

# 额外校验：lidar 的 H/W 也应该是 10 倍缩放（如果 lidar 与 hsi_pca 尺度对齐）
try:
    lidar_arr = _to_numpy(lidar)
    lidar_h, lidar_w = int(lidar_arr.shape[-2]), int(lidar_arr.shape[-1])
    lidar_ratio_h = rgb_h / max(lidar_h, 1)
    lidar_ratio_w = rgb_w / max(lidar_w, 1)
    print(
        f"[shape-check] ref(lidar) H,W=({lidar_h},{lidar_w}) -> rgb H,W=({rgb_h},{rgb_w}) "
        f"ratio=(H:{lidar_ratio_h:.3f}, W:{lidar_ratio_w:.3f})"
    )
    if abs(lidar_ratio_h - 10.0) > 1e-3 or abs(lidar_ratio_w - 10.0) > 1e-3:
        print("[shape-check][WARN] rgb 相对 lidar 的 H/W 未严格等于 10 倍。")
except Exception:
    pass

# visualize_hsi_lidar_data(hsi_pca, lidar, train, test, rgb=rgb_chw)
houston2018 = {
    'hsi': hsi_pca,
    'lidar': lidar,
    'rgb': rgb_chw,
    'train': train,
    'test': test
}
import scipy.io as sio
os.makedirs('../FusAtNet/data2/', exist_ok=True)
sio.savemat('../FusAtNet/data2/houston2018.mat', houston2018)
print()