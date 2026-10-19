# 数据准备计划（多轮对话共同维护）

> 用途：记录从 3DReasonKnee DESS 膝关节 MRI 整理成 nnU-Net v2 训练数据的过程与后续计划。
> 每次对话完成后在此追加/更新进行的主要操作，保持与最新状态一致。

## 目标

把本地已有数据整理成 nnU-Net v2 可直接训练的数据集（`nnUNet_raw/DatasetXXX_Name`），
并完成图像/标签一致性校验。

## 数据来源与目录

| 目录/文件 | 说明 |
|---|---|
| `dess_nii/` | 原始 DESS 图像，100 个，文件名为 DICOM UID（`.nii.gz`） |
| `dess_basic/` | 标签 mask，100 个，结构为 `<case_id>/<series>.nii.gz` |
| `metadata/segmentation_index.csv` | 标签索引（case_id/series → mask 路径、seg__/sub__ 标志、mask_label_map） |
| `metadata/metadata.csv` | 患者/研究元数据（case_id/series → dess_uid、dess_nii 路径等） |
| `metadata/merged_index.csv` | **合并后的索引表（产物）** |

- 两表以 `(case_id, series)` 为键，**1:1 完全对应**（各 100 行，仅排序不同）。
- `case_id` 在 100 例中唯一，可作为 nnU-Net 的 case 标识符。
- `series` 取值：`A B C D E L N X Y`；`knee_side` 取值：`1 2`。

## 已完成操作

1. **合并索引表 + 修正路径** → `metadata/merged_index.csv`（100 行，111 列）
   - 修正 `dess_nii` → `/root/autodl-tmp/nnUNet/dess_nii/<uid>.nii.gz`
   - 修正 `mask_path` → `/root/autodl-tmp/nnUNet/dess_basic/<case_id>/<series>.nii.gz`
   - 两表重叠的 `n_segmentations`/`n_subregions` 数值不同，分别保留为
     `n_segmentations`/`n_subregions`（metadata）与 `mask_n_segmentations`/`mask_n_subregions`（segmentation_index）。

2. **图像/标签一致性校验（已标注进 CSV）**
   - 新增列：`image_shape`、`label_shape`、`shape_match`、`image_spacing`、`label_spacing`、`spacing_match`、`image_origin`、`label_origin`、`origin_match`、`present_labels`、`label_check`、`exclude_reason`
   - 形状：100/100 匹配（全部 `(160, 384, 384)`）
   - spacing：100/100 匹配（`(0.7, 0.364583, 0.364583)`）
   - origin（sform 平移量，容差 0.001mm）：99/100 匹配；1 例真实不一致（`9454239-20080721/A`，偏移约 162mm）
   - label 值：100/100 合法，无越界

3. **生成 nnU-Net v2 数据集** → `nnUNet_raw/Dataset901_KneeDESS/`
   - `imagesTr/<case_id>_0000.nii.gz`（97 个，硬链接到 dess_nii，不重复占磁盘）
   - `labelsTr/<case_id>.nii.gz`（97 个，重映射后的标签，uint16）
   - `dataset.json`（单通道 DESS，12 类 + background，`numTraining=97`，`file_ending=".nii.gz"`）

## 标签映射（重要，重映射为连续值）

原始 mask 标签值非连续，nnU-Net 强制要求连续 `0..N`，已重映射：

| 新值 | 结构 | 旧值 |
|---|---|---|
| 0 | background | 0 |
| 1 | acl | 1 |
| 2 | femur | 3 |
| 3 | femur_cartilage | 4 |
| 4 | hoffa_synovitis | 23 |
| 5 | lateral_meniscus | 24 |
| 6 | medial_meniscus | 28 |
| 7 | patella | 32 |
| 8 | patella_cartilage | 33 |
| 9 | patellar_tendon | 42 |
| 10 | pcl | 43 |
| 11 | tibia | 44 |
| 12 | tibia_cartilage | 45 |

- 排除后 12 类在全部 97 例中都存在（原 100 例中 hoffa 在 2 例为 null，这 2 例已排除）。

## 已排除样本（不进入 Dataset901）

| case_id | 原因 |
|---|---|
| `-10300106/A` | `hoffa_synovitis` 标签为 null |
| `-9000798-20060303/E` | `hoffa_synovitis` 标签为 null |
| `9454239-20080721/A` | 图像/label origin 不一致（偏移约 162mm） |

## 待办 / 下一步

- [ ] 安装 `nnunetv2`（`pip install nnunetv2`，会带上 nibabel 等；当前环境只有 numpy/torch）
- [ ] 设置环境变量：`nnUNet_raw=/root/autodl-tmp/nnUNet/nnUNet_raw`（`nnUNet_preprocessed`、`nnUNet_results` 同理）
- [ ] 运行 `nnUNetv2_verify_dataset_integrity Dataset901_KneeDESS` 二次校验
- [ ] 运行 `nnUNetv2_plan_and_preprocess Dataset901_KneeDESS ...` 生成计划与预处理数据
- [ ] 训练 / 划分（如需按 patient 划分避免同患者不同 knee 泄漏，注意 metadata 中 `patient_id` 复用）

## 关键决策与注意

- 数据集 ID/名称暂定 `Dataset901_KneeDESS`（避开 MSD 占用的 001–010），可随时改名（改文件夹名 + dataset.json）。
- 标签保留 `uint16`（重映射未改 NIfTI 头，最稳妥）；如需省空间可转 `uint8`（值域 0–12 足够）。
- 图片用硬链接而非复制，避免重复占 2.5G 磁盘。
- 本次形状校验是直接解析 NIfTI 头完成的（未依赖 nibabel），结果与 nnU-Net 的 `check_cases` 一致（spacing 用 allclose 容差）。
