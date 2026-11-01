#!/usr/bin/env python3
"""为一个 DESS 验证 fold 准备推理输入（DESS 或 TSE）。

读取 workspace/inference/config.yaml（可用 --config 指定其他路径），
根据其中的 dataset_id、fold 与 modality（dess / tse），脚本会：
  1. 从 nnUNet_preprocessed/Dataset{id}_*/splits_final.json 取出该 fold 的验证集 case（dess_XXX）；
  2. 经 conversion_report_Dataset{id}.json 把 case_id 映射回 DESS 影像的 DICOM UID（dess_uid）；
  3. 决定要复制哪类影像：
       - modality: dess -> 直接复制 raw_data/dess_nii/{dess_uid}.nii.gz；
       - modality: tse  -> 经 metadata.csv 的 dess_uid -> tse_uid 对应关系找到 TSE 影像，
                           复制 raw_data/tse_nii/{tse_uid}.nii.gz；
  4. 把命中的文件复制到 workspace/inference/{dataset_id}_{fold}_{modality}/input/，
     另外复制到 input_unet/ 并补上 nnU-Net 的 _0000 通道后缀（供 nnUNetv2_predict 直接使用），
     同时在旁边创建空的 output/ 目录，并写出 mapping.json 记录 case -> 影像 的对应关系。

tse 模式：metadata.csv 优先；若某 case 的 dess_uid 不在 metadata 里，则退回「末尾两位不同、
其余相同」的 UID 规则兜底。两种都找不到的 case 会原样报告，不会被静默跳过或猜测配对。

用法：
    python prepare_inference.py                    # 读取同目录 config.yaml
    python prepare_inference.py --config other.yaml
"""

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

FILE_ENDING = ".nii.gz"
TSE_UID_SUFFIX_CHARS = 2  # tse 兜底规则：末尾两位（'.' + 序列号数字）为核磁序列标识
NNUNET_CHANNEL_SUFFIX = "_0000"  # nnU-Net 单通道输入的通道后缀，input_unet/ 里会补上

# 表示「成功找到并复制了源影像」的状态；dess 模式用 found，tse 模式用 metadata/uid_rule。
SUCCESS_STATUSES = {"found", "metadata", "uid_rule"}


def resolve_repo_root():
    """返回仓库根目录（本脚本位于 <root>/workspace/inference/ 下）。"""
    return Path(__file__).resolve().parents[2]


def find_single_dir(pattern, what):
    matches = sorted(pattern.parent.glob(pattern.name))
    if len(matches) == 0:
        sys.exit(f"找不到 {what}：{pattern}")
    if len(matches) > 1:
        sys.exit(f"{what} 不唯一，匹配到多个：{[m.name for m in matches]}")
    return matches[0]


def load_conversion_report(repo_root, dataset_id):
    """读转换报告，返回 {case_id: dess_uid}（已去掉 .nii.gz 后缀）。"""
    path = (
        repo_root / "workspace" / "dess2unetdataset"
        / f"conversion_report_Dataset{dataset_id:03d}.json"
    )
    if not path.is_file():
        sys.exit(f"找不到转换报告：{path}")
    report = json.loads(path.read_text())
    cases = report.get("cases")
    if not cases:
        sys.exit(f"转换报告 {path} 中没有 'cases' 字段")
    mapping = {}
    for c in cases:
        name = c["original_image"]
        mapping[c["case_id"]] = name[: -len(FILE_ENDING)] if name.endswith(FILE_ENDING) else name
    return mapping


def load_validation_cases(repo_root, dataset_id, fold):
    """从 splits_final.json 读取指定 fold 的验证集 case 列表。"""
    preprocessed = repo_root / "nnUNet_preprocessed"
    dataset_dir = find_single_dir(
        preprocessed / f"Dataset{dataset_id:03d}_*", "预处理数据集目录"
    )
    splits_path = dataset_dir / "splits_final.json"
    if not splits_path.is_file():
        sys.exit(f"找不到 splits_final.json：{splits_path}")
    splits = json.loads(splits_path.read_text())
    if not isinstance(splits, list) or fold >= len(splits):
        sys.exit(
            f"fold {fold} 超出范围（splits_final.json 共 {len(splits) if isinstance(splits, list) else '?'} 个 fold）"
        )
    return splits[fold]["val"]


def load_metadata_mapping(repo_root):
    """读 metadata.csv，返回 {dess_uid: tse_uid}。"""
    path = repo_root / "workspace" / "inference" / "metadata.csv"
    if not path.is_file():
        sys.exit(f"找不到 metadata.csv：{path}")
    mapping = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["dess_uid"]] = row["tse_uid"]
    return mapping


def build_image_index(repo_root, modality):
    """返回 {去除 .nii.gz 的 UID: 文件名}，来自 raw_data/dess_nii 或 raw_data/tse_nii。"""
    subdir = "dess_nii" if modality == "dess" else "tse_nii"
    img_dir = repo_root / "raw_data" / subdir
    if not img_dir.is_dir():
        sys.exit(f"找不到影像目录：{img_dir}")
    index = {}
    for p in sorted(img_dir.glob(f"*{FILE_ENDING}")):
        stem = p.name[: -len(FILE_ENDING)]
        index[stem] = p.name
    return index


def find_tse(dess_uid, metadata_map, tse_stems):
    """返回 (状态, tse_uid 或 None)。

    状态：metadata / uid_rule / ambiguous / missing。
    """
    tse_uid = metadata_map.get(dess_uid)
    if tse_uid and tse_uid in tse_stems:
        return "metadata", tse_uid

    key = dess_uid[:-TSE_UID_SUFFIX_CHARS]
    candidates = [t for t in tse_stems if t[:-TSE_UID_SUFFIX_CHARS] == key]
    if len(candidates) == 1:
        return "uid_rule", candidates[0]
    if len(candidates) > 1:
        return "ambiguous", None
    return "missing", None


def load_config(config_path):
    """读 YAML 配置，返回 (dataset_id, fold, modality)。"""
    cfg = yaml.safe_load(config_path.read_text())
    try:
        dataset_id = int(cfg["dataset_id"])
        fold = int(cfg["fold"])
    except (KeyError, TypeError) as e:
        sys.exit(f"配置 {config_path} 缺少 dataset_id / fold 字段：{e}")
    modality = str(cfg.get("modality", "tse")).lower()
    if modality not in ("dess", "tse"):
        sys.exit(f"配置 {config_path} 的 modality 必须是 'dess' 或 'tse'，当前为 {modality!r}")
    return dataset_id, fold, modality


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None,
                        help="path to the YAML config (default: config.yaml next to this script)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config) if args.config else script_dir / "config.yaml"
    if not config_path.is_file():
        sys.exit(f"config not found: {config_path}")
    dataset_id, fold, modality = load_config(config_path)

    repo_root = resolve_repo_root()
    case_to_dess = load_conversion_report(repo_root, dataset_id)
    val_cases = load_validation_cases(repo_root, dataset_id, fold)

    metadata_map = load_metadata_mapping(repo_root) if modality == "tse" else {}
    image_index = build_image_index(repo_root, modality)
    stems = set(image_index.keys())
    src_dir = repo_root / "raw_data" / ("dess_nii" if modality == "dess" else "tse_nii")

    out_dir = repo_root / "workspace" / "inference" / f"{dataset_id}_{fold}_{modality}"
    input_dir = out_dir / "input"
    input_unet_dir = out_dir / "input_unet"
    output_dir = out_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_unet_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    stats = defaultdict(int)

    for case_id in sorted(val_cases):
        dess_uid = case_to_dess.get(case_id)
        if dess_uid is None:
            print(f"[跳过] {case_id}: 转换报告中没有该 case 的映射")
            stats["no_case"] += 1
            records.append({"case_id": case_id, "modality": modality, "status": "no_case"})
            continue

        if modality == "dess":
            status = "found" if dess_uid in stems else "missing"
            source_uid = dess_uid if status == "found" else None
        else:
            status, source_uid = find_tse(dess_uid, metadata_map, stems)

        record = {"case_id": case_id, "dess_uid": dess_uid,
                  "modality": modality, "status": status}
        if status in SUCCESS_STATUSES:
            source_file = image_index[source_uid]
            source_file_unet = source_file[: -len(FILE_ENDING)] + NNUNET_CHANNEL_SUFFIX + FILE_ENDING
            shutil.copy2(src_dir / source_file, input_dir / source_file)
            shutil.copy2(src_dir / source_file, input_unet_dir / source_file_unet)
            record.update({"source_uid": source_uid, "source_file": source_file,
                           "source_file_unet": source_file_unet})
            print(f"[复制] {case_id}  {dess_uid}  ->  {source_file}  ({status})")
        elif status == "ambiguous":
            print(f"[警告] {case_id}  {dess_uid}  匹配到多个 {modality.upper()} 候选")
        else:
            print(f"[缺失] {case_id}  {dess_uid}  未找到对应 {modality.upper()} 文件")
        stats[status] += 1
        records.append(record)

    (out_dir / "mapping.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False)
    )

    copied = sum(stats[s] for s in SUCCESS_STATUSES)
    print()
    print("================ 汇总 ================")
    print(f"验证集 case 总数 : {len(val_cases)}")
    print(f"影像类型        : {modality}")
    if modality == "tse":
        print(f"按 metadata 复制 : {stats['metadata']}")
        print(f"按 UID 规则复制 : {stats['uid_rule']}")
    print(f"成功复制        : {copied}")
    print(f"匹配到多个      : {stats['ambiguous']}")
    print(f"未找到文件      : {stats['missing']}")
    print(f"无 case 映射    : {stats['no_case']}")
    print(f"输入目录        : {input_dir}")
    print(f"输入目录(nnU)   : {input_unet_dir}")
    print(f"输出目录        : {output_dir}")


if __name__ == "__main__":
    main()
