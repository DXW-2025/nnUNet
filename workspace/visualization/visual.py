"""将分割标签叠加到影像上，输出切片拼图。

支持两种模式（二选一）：
- 单张：配置 image / mask 两个文件路径，输出单个拼图。
- 批量：配置 image_dir / mask_dir 两个文件夹，逐张输出拼图到 output 目录。
  配对规则：影像的相对路径（或文件名）与标签一一对应；若影像在 mask_dir 下
  不存在同名相对路径，则退回按文件名唯一匹配。

标签默认仅用边框（轮廓）描述，不填充；可通过 mask_style: "fill" 切换为半透明填充。
用法：python visual.py [--config path/to/config.yaml] [--image-dir ... --mask-dir ...]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")

NIFTI_EXTS = (".nii.gz", ".nii")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="叠加分割标签并输出切片拼图")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    # 命令行参数可覆盖配置文件中的同名路径。
    parser.add_argument("--image", type=Path, help="单张影像路径")
    parser.add_argument("--mask", type=Path, help="单张标签路径")
    parser.add_argument("--image-dir", type=Path, help="批量影像文件夹")
    parser.add_argument("--mask-dir", type=Path, help="批量标签文件夹")
    parser.add_argument("--output", type=Path, help="输出文件（单张）或目录（批量）")
    return parser.parse_args(argv)


def is_nifti(path):
    return path.is_file() and any(path.name.endswith(ext) for ext in NIFTI_EXTS)


def is_under(path, directory):
    """path 是否位于 directory 目录内（含其子目录）。

    先 resolve 再比较，因此无论 path / directory 传入的是绝对路径还是相对路径
    （乃至两者混用），都能正确判断。
    """
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
        return True
    except ValueError:
        return False


def strip_nifti_ext(path):
    """去掉 .nii.gz / .nii 后缀，返回文件名主体。"""
    name = path.name
    for ext in NIFTI_EXTS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return path.stem


# 路径两侧可能误带的成对引号（含直引号与中文/弯引号）。
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"))


def strip_quotes(value):
    """去掉字符串两侧误带的成对引号，避免绝对路径粘贴时带上引号导致找不到文件。"""
    text = str(value).strip()
    for left, right in _QUOTE_PAIRS:
        if len(text) >= 2 and text.startswith(left) and text.endswith(right):
            return text[1:-1].strip()
    return text


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("配置必须为 YAML 字典")

    def resolve(value):
        if value is None:
            return None
        value = Path(strip_quotes(value)).expanduser()
        return (value if value.is_absolute() else path.parent / value).resolve()

    for key in ("image", "mask", "output", "image_dir", "mask_dir"):
        config[key] = resolve(config.get(key))

    alpha = config.get("mask_alpha", 0.6)
    if not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1:
        raise ValueError("mask_alpha 必须是 0~1 之间的数值")
    config["mask_alpha"] = float(alpha)

    style = config.get("mask_style", "contour")
    if style not in ("contour", "fill"):
        raise ValueError('mask_style 必须是 "contour"（边框）或 "fill"（填充）')
    config["mask_style"] = style

    config.setdefault("mask_color", "auto")

    n_col = config.get("n_col", 4)
    if not isinstance(n_col, int) or n_col < 1:
        raise ValueError("n_col 必须是正整数")
    config["n_col"] = int(n_col)

    window = config.get("window", [1, 99])
    if (not isinstance(window, (list, tuple)) or len(window) != 2
            or not all(isinstance(v, (int, float)) for v in window)
            or not 0 <= window[0] <= 100 or not 0 <= window[1] <= 100
            or window[0] >= window[1]):
        raise ValueError("window 必须是两个 0~100 的百分位数，且第一个小于第二个，如 [1, 99]")
    config["window"] = [float(window[0]), float(window[1])]

    vmin = config.get("vmin")
    vmax = config.get("vmax")
    if (vmin is None) != (vmax is None):
        raise ValueError("vmin 与 vmax 必须同时提供（固定窗位），或都不提供（用 window 百分位）")
    if vmin is not None:
        if not isinstance(vmin, (int, float)) or not isinstance(vmax, (int, float)):
            raise ValueError("vmin / vmax 必须是数值")
        config["vmin"] = float(vmin)
        config["vmax"] = float(vmax)
    else:
        config["vmin"] = None
        config["vmax"] = None

    config["contour_width"] = float(config.get("contour_width", 1.0))
    return config


def resolve_colors(mask_color, n_classes):
    """根据配置返回长度 n_classes 的颜色列表。"""
    if isinstance(mask_color, str) and mask_color == "auto":
        return [list(c) for c in plt.cm.tab20(np.linspace(0, 1, n_classes))]

    if isinstance(mask_color, str):
        # 单一颜色：所有类别同色。
        return [mask_color] * n_classes

    if isinstance(mask_color, (list, tuple)):
        if len(mask_color) != n_classes:
            raise ValueError(
                f"mask_color 列表长度 {len(mask_color)} 与类别数 {n_classes} 不一致"
            )
        return list(mask_color)

    raise ValueError('mask_color 必须是 "auto"、单一颜色或颜色列表')


def compute_window(data, config):
    """计算统一显示窗口 (vmin, vmax)，供所有切片共用。

    优先使用固定的 vmin/vmax（绝对窗宽窗位）；未提供则按 window 百分位计算。
    """
    if config["vmin"] is not None:
        vmin, vmax = config["vmin"], config["vmax"]
    else:
        vmin, vmax = np.percentile(data, config["window"])
    if vmax <= vmin:
        vmax = vmin + 1.0  # 灰度恒定时避免 vmin == vmax 触发 imshow 报错
    return float(vmin), float(vmax)


def draw_slice(ax, data_slice, mask_slice, classes, colors, config, vmin, vmax):
    """在 ax 上绘制一帧灰底影像并叠加标签。vmin/vmax 为统一显示窗口。"""
    ax.imshow(np.fliplr(data_slice.T), cmap="gray", origin="lower", vmin=vmin, vmax=vmax)

    if config["mask_style"] == "fill":
        # 把任意类别值重排为 1..n，便于 ListedColormap 按序取色。
        ranked = np.zeros_like(mask_slice, dtype=np.float64)
        for rank, value in enumerate(classes, start=1):
            ranked[mask_slice == value] = rank
        ranked = np.ma.masked_where(ranked == 0, ranked)
        cmap = ListedColormap(colors)
        vmin, vmax = (0, 1) if len(classes) == 1 else (1, len(classes))
        ax.imshow(
            np.fliplr(ranked.T),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=config["mask_alpha"],
            interpolation="nearest",
            origin="lower",
        )
        return

    # contour 模式：逐类绘制边框。
    for value, color in zip(classes, colors):
        binary = mask_slice == value
        if not binary.any():
            continue
        ax.contour(
            np.fliplr(binary.T),
            levels=[0.5],
            colors=[color],
            linewidths=config["contour_width"],
            origin="lower",
        )


def render_volume(data, mask, config, title=None):
    classes = np.unique(mask[mask != 0])
    n_classes = len(classes)
    if n_classes == 0:
        raise ValueError("mask 中没有任何非零标签")
    print(f"类别值 ({title or ''}):", classes)

    colors = resolve_colors(config["mask_color"], n_classes)
    vmin, vmax = compute_window(data, config)

    n_slice = data.shape[0]
    n_col = config["n_col"]
    n_row = n_slice // n_col + (n_slice % n_col > 0)
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 2.5, n_row * 2.5))
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_slice):
        ax = axes[i]
        draw_slice(ax, data[i, :, :], mask[i, :, :], classes, colors, config, vmin, vmax)
        ax.set_title(f"Slice {i}", fontsize=8)
        ax.axis("off")

    for i in range(n_slice, len(axes)):
        axes[i].axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=10)

    plt.tight_layout()
    return fig


def load_pair(image_path, mask_path):
    data = nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    if data.shape != mask.shape:
        raise ValueError(
            f"影像与标签尺寸不一致：{image_path} {data.shape} vs {mask_path} {mask.shape}"
        )
    return data, mask


def run_single(config):
    data, mask = load_pair(config["image"], config["mask"])
    fig = render_volume(data, mask, config)

    if config["output"] is not None:
        fig.savefig(config["output"], bbox_inches="tight", dpi=150)
        print("已保存:", config["output"])
    else:
        plt.show()
    plt.close(fig)


def run_batch(config):
    image_dir = config["image_dir"]
    mask_dir = config["mask_dir"]

    # 建立标签索引：文件名 -> 候选路径（供唯一匹配回退）。
    mask_index = {}
    for path in mask_dir.rglob("*"):
        if is_nifti(path):
            mask_index.setdefault(path.name, []).append(path)

    images = sorted(
        path
        for path in image_dir.rglob("*")
        if is_nifti(path) and not is_under(path, mask_dir)
    )
    if not images:
        raise ValueError(f"未在 {image_dir} 中找到任何影像文件（*.nii.gz / *.nii）")

    output_dir = config["output"]
    n_total = len(images)
    n_done = 0

    for img in images:
        rel = img.relative_to(image_dir)

        # 1) 优先按相对路径匹配；2) 退回按文件名唯一匹配。
        mask = mask_dir / rel
        if not mask.is_file():
            candidates = mask_index.get(img.name, [])
            if len(candidates) == 1:
                mask = candidates[0]
            else:
                print(f"跳过 {rel}：标签不存在或文件名无法唯一匹配")
                continue

        data, m = load_pair(img, mask)
        fig = render_volume(data, m, config, title=str(rel))

        out_path = output_dir / f"{rel.with_name(strip_nifti_ext(rel))}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        n_done += 1
        print(f"[{n_done}/{n_total}] 已保存:", out_path)

    print(f"完成：共处理 {n_done}/{n_total} 张影像")


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)

    # 命令行参数覆盖配置文件中的路径（相对路径以当前工作目录为基准）。
    for key, value in (
        ("image", args.image),
        ("mask", args.mask),
        ("image_dir", args.image_dir),
        ("mask_dir", args.mask_dir),
        ("output", args.output),
    ):
        if value is not None:
            config[key] = Path(strip_quotes(value)).expanduser().resolve()

    single = config.get("image") is not None and config.get("mask") is not None
    batch = config.get("image_dir") is not None and config.get("mask_dir") is not None
    if single and batch:
        raise ValueError("(image, mask) 与 (image_dir, mask_dir) 只能配置一组")
    if not single and not batch:
        raise ValueError("请配置 (image, mask) 或 (image_dir, mask_dir) 其中一组")

    if batch:
        if config.get("output") is None:
            config["output"] = args.config.expanduser().resolve().parent / "visualization"
        config["output"].mkdir(parents=True, exist_ok=True)
        run_batch(config)
    else:
        if config.get("output") is not None:
            config["output"].parent.mkdir(parents=True, exist_ok=True)
        run_single(config)


if __name__ == "__main__":
    main()
