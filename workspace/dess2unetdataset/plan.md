# plan.md — DESS → nnU-Net 数据转换与处理

本文件记录数据从原始 NIfTI 到 nnU-Net v2 数据集的完整转换与处理逻辑。

## 目的

把 DESS 膝关节 MRI 图像（`raw_data/dess_nii`）+ BML（骨髓水肿）标签（`raw_data/dess_bml`）
转换为符合 nnU-Net v2 规范的数据集，供后续训练使用。

## 数据概况

| 输入 | 路径 | 数量 | 说明 |
|---|---|---|---|
| 图像 | `raw_data/dess_nii` | 100 | DESS 单通道，以 DICOM UID 命名 |
| 标签 | `raw_data/dess_bml` | 100 | BML 标签，以 DICOM UID 命名，另含 `bml_label_map.json` |

关键事实：图像与标签**均以 DICOM UID 命名，但两套 UID 不一一对应**，无法按文件名配对。
标签与其来源图像处于同一物理空间，因此改用 NIfTI 的 **origin（`affine[:3,3]`，即平移分量）** 作为配对键。

`bml_label_map.json` 是 `{name: value}` 形式的标签名映射，共 17 项：3 个解剖结构
（femur=3、patella=32、tibia=44）+ 14 个 BML 子区域。原始标签值**非连续**
（0,3,6,9,12,15,18,21,32,35,38,44,46,47,50,51,53,56），需要重映射为连续的 0..17。

## 转换脚本 `convert_dess_to_nnunet.py` 的处理逻辑

### 第 1 步：读取 YAML 配置

`config.yaml` 指定输入/输出路径与行为参数：

- `image_folder` / `label_folder`：输入图像与标签文件夹
- `output_raw_folder` / `dataset_id` / `dataset_name`：输出位置与数据集标识
- `channel_names`：通道名（`{0: DESS}`，非 CT 名 → nnU-Net 使用逐通道 z-score 归一化）
- `origin_tolerance_mm`：图像/标签 origin 配对容差（0.5 mm）
- `case_id_prefix` / `file_ending`：输出文件名前缀与格式（`dess` / `.nii.gz`）
- `overwrite_image_reader_writer`：可强制指定 nnU-Net 读写器（默认 null，自动探测）
- `label_map_json`：标签名映射 JSON 文件名（相对 label_folder 或绝对路径）
- `label_names`：手动覆盖 `{原始值: 名称}`（默认空）
- `label_merges`：可选的标签合并规则（见第 5 步）
- `exclude_if_empty`：可选的"指定原始结构为空则排除该 case"规则（见第 3 步）
- `license`：写入 dataset.json 的元数据

### 第 2 步：按 origin 配对图像与标签，排除不一致者

1. 扫描两个文件夹中所有 `*.nii.gz`，读取各自 NIfTI 的 origin（`affine[:3,3]`）。
2. 用容差 `origin_tolerance_mm` 判定两个 origin 是否一致（逐分量差 ≤ 容差）。
3. 图像与标签按 origin 互相配对：
   - 有图像无匹配标签 → 图像列入 `excluded_images`
   - 有标签无匹配图像 → 标签列入 `excluded_labels`
4. 对每个配对再校验几何一致性：shape 与 spacing 必须一致（不一致即排除，记入 `invalid_pairs`）。

本数据集结果：100 例中 1 例被排除（图像 `...1218148065.1.0` 与标签 `...1218148065.1.0.2`
origin 不同），保留 99 例。

### 第 3 步：排除指定原始结构为空的 case（可选）

`exclude_if_empty` 列出必须非空的**原始标签**：可用 label-map JSON 中的名称，或直接写原始整数值。
在 origin 配对与几何校验之后、标签合并之前，对每个保留 case：

- 检查该 case 标签中是否包含所有列出的原始结构；
- 任一列出结构在该 case 中无任何体素（为空）→ 该 case 被排除，记入 `excluded_empty`。

排除完成后，**类值并集只在存活 case 上重新计算**，避免被排除 case 中独有的标签残留到最终类表。

### 第 4 步：读取标签名映射（label-map JSON）

- 从 `label_map_json` 指定的 JSON（或自动发现 label_folder 内唯一的 `*.json`）读取 `{name: value}`。
- 在 `label_names` 中未提供名称的原始值，回退为 `label_<value>`。

### 第 5 步：可选的标签合并（label_merges）

`label_merges` 是 `[{name, labels}, ...]` 列表，用于把多个标签合并为一个新类。校验规则：

- 每个 `labels` 中的名称必须存在于 label-map JSON，否则报错退出；
- 名称不能重复，且同一成员不能出现在多个合并组。

`name: background` 是特殊合并目标：其成员被并入 background（映射为 0，不再作为独立类别）。

合并后的类在报告中记录其新值与原始值集合。

### 第 6 步：构建标签映射并重映射为连续值

1. **类集合来自数据本身**：遍历所有存活（保留）标签，取实际出现过的原始值的并集。
   仅 JSON 声明但数据中从未出现的标签**不会进入最终类表**（静默丢弃）。
2. 若存在合并规则，把属于该合并的原始值统一映射到同一个新值。
3. 剩余值按最小值排序，重映射为连续的 `0..N`（0 恒为 background）。
4. 对每个标签逐体素写入重映射后的结果。

### 第 7 步：写入 nnU-Net 数据集

生成标准目录结构：

```
nnUNet_raw/Dataset902_KneeDESS_BML/
├── imagesTr/{case_id}_0000.nii.gz   # 单通道图像
├── labelsTr/{case_id}.nii.gz        # 重映射后的标签
└── dataset.json                      # 通道、类别名、数量、许可证等
```

`dataset.json` 用 `nnunetv2.dataset_conversion.generate_dataset_json` 生成，
`labels` 键记录了连续的类别名（background 开头）。

### 第 8 步：生成转换报告并验证

报告 `workspace/conversion_report_Dataset902.json` 记录：

- `label_mapping`：原始值 → 新值 → 名称的完整映射
- `label_merges`：合并类的名称、新值与原始值集合
- `label_distribution`：合并后每个类出现的 case 数（见下方统计规则）
- `excluded_images` / `excluded_labels` / `invalid_pairs` / `excluded_empty`：被排除的项及原因
- `cases`：保留的 case 列表

**合并后类别分布统计规则**：对每个 case，取该 case 内所有原始值经映射后的新值集合
（**去重后**计数）。这样当一个 case 内多个原始值映射到同一合并类时，合并类只计 1 次，
避免重复计数。

最后调用 `nnunetv2.experiment_planning.verify_dataset_integrity.verify_dataset_integrity`
校验：标签仅含声明值（缺失类别允许，出现未声明值则报错）、shape/spacing 一致（硬性）、
affine/origin/direction 不一致（仅警告）。

## 结果

- 输出数据集：`nnUNet_raw/Dataset902_KneeDESS_BML`（99 训练例，单通道 DESS，18 类含 background）。
- `verify_dataset_integrity` 通过。
- 类别分布（合并前，new value → case 数）：femur/patella/tibia 各 99，BML 子区域 2～38 例不等。
- 完整记录见 `workspace/conversion_report_Dataset902.json`。

## 运行方式

```bash
/root/miniconda3/envs/unet/bin/python workspace/convert_dess_to_nnunet.py --config workspace/config.yaml
```

改动配置（如 dataset_id、label_merges、exclude_if_empty）后重新运行即可生成对应数据集。
