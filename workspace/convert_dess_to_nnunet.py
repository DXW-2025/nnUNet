#!/usr/bin/env python3
"""Convert DESS knee MR images + BML labels into an nnU-Net v2 raw dataset.

Why pairing is done by spatial origin
-------------------------------------
The source images and labels are both named by DICOM SOP-instance UIDs, but the
UID sets of the two folders do not correspond 1:1 (the label was exported in a
separate step and received its own UIDs). The only reliable key that ties a
label to its image is the spatial origin stored in the NIfTI affine (the
translation component ``affine[:3, 3]``): a label and its image live in the
same physical space, so they share that origin. This script therefore

  1. reads a YAML config pointing at the image folder and the label folder,
  2. pairs every image with its label by matching their origin,
  3. validates each pair's geometry (origin, shape, spacing) and *excludes*
     any case whose label origin does not match its image,
  4. remaps the non-consecutive source label values to consecutive 1..N
     (0 stays background), which nnU-Net requires,
  5. writes a standard nnU-Net v2 dataset:

        DatasetXXX_NAME/
        ├── dataset.json
        ├── imagesTr/{case_id}_0000.nii.gz
        └── labelsTr/{case_id}.nii.gz

Class names are read from a ``{name: value}`` label-map JSON inside the label
folder (auto-discovered, or picked via ``label_map_json`` in the config);
``label_names`` in the config can still override individual entries, and
``label_merges`` can fuse several labels into one class under a new name (each
referenced label is validated against the label-map JSON). Values missing from
the map fall back to "label_<orig>". The full mapping, the merges, and the list
of excluded cases are written to a JSON report next to this script.

Usage
-----
    /root/miniconda3/envs/unet/bin/python convert_dess_to_nnunet.py --config config.yaml
    /root/miniconda3/envs/unet/bin/python convert_dess_to_nnunet.py --config config.yaml --overwrite
    /root/miniconda3/envs/unet/bin/python convert_dess_to_nnunet.py --config config.yaml --verify
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def read_geometry(path):
    """Return origin, shape and spacing of a NIfTI file (header only + shape)."""
    img = nib.load(path)
    affine = np.asarray(img.affine, dtype=np.float64)
    return {
        "path": str(path),
        "name": os.path.basename(path),
        "origin": affine[:3, 3],                                   # physical origin (mm)
        "shape": tuple(img.shape[:3]),
        "zooms": np.asarray(img.header.get_zooms()[:3], dtype=np.float64),
    }


def origins_close(a, b, tol):
    return bool(np.allclose(a, b, atol=tol))


def geometry_is_consistent(image_geom, label_geom, tol):
    """Return (ok, reason). Origin is the pairing key; shape/spacing are the
    hard requirements nnU-Net itself enforces, so a mismatch excludes the case."""
    if not origins_close(image_geom["origin"], label_geom["origin"], tol):
        return False, "origin mismatch"
    if image_geom["shape"] != label_geom["shape"]:
        return False, "shape mismatch"
    if not np.allclose(image_geom["zooms"], label_geom["zooms"], atol=1e-3):
        return False, "spacing mismatch"
    return True, ""


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #

def pair_by_origin(image_geoms, label_geoms, tol):
    """Match each image to exactly one label by origin (bijection)."""
    matched = []
    excluded_images = []
    used_label_idx = set()

    for ig in image_geoms:
        candidates = [
            (j, lg) for j, lg in enumerate(label_geoms)
            if j not in used_label_idx and origins_close(ig["origin"], lg["origin"], tol)
        ]
        if len(candidates) == 1:
            matched.append((ig, candidates[0][1]))
            used_label_idx.add(candidates[0][0])
        elif len(candidates) == 0:
            excluded_images.append((ig, "no label with a matching origin"))
        else:
            excluded_images.append(
                (ig, f"ambiguous origin: {len(candidates)} labels share this origin"))

    excluded_labels = [
        (lg, "no image with a matching origin")
        for j, lg in enumerate(label_geoms) if j not in used_label_idx
    ]
    return matched, excluded_images, excluded_labels


# --------------------------------------------------------------------------- #
# Label value handling
# --------------------------------------------------------------------------- #

def collect_label_values(label_paths):
    """Return (sorted union of original values, per-case set of original values)."""
    values = set()
    per_case = []
    for p in label_paths:
        data = np.asanyarray(nib.load(p).dataobj)
        uniq = set(np.unique(data).astype(int).tolist())
        values.update(uniq)
        per_case.append(uniq)
    return sorted(values), per_case


def load_label_map_json(label_folder, cfg):
    """Load a ``{name: value}`` label-map JSON from the label folder.

    Resolution order:
      1. ``label_map_json`` from the config (a filename relative to the label
         folder, or an absolute path) if provided;
      2. otherwise the single ``*.json`` file found in the label folder.

    Returns a ``{name: value}`` dict; returns ``{}`` (with a warning) when no
    usable map is found, in which case classes fall back to "label_<value>".
    """
    explicit = cfg.get("label_map_json")
    candidates = []
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = label_folder / path
        if path.is_file():
            candidates = [path]
        else:
            print(f"[warn] label_map_json '{explicit}' not found; falling back to auto names")
    else:
        candidates = sorted(label_folder.glob("*.json"))

    if len(candidates) == 0:
        print("[warn] no label-map JSON in label folder; classes will be named 'label_<value>'")
        return {}
    if len(candidates) > 1:
        print(f"[warn] multiple JSON files in label folder: {[c.name for c in candidates]}; "
              "set 'label_map_json' in the config to pick one. Falling back to auto names.")
        return {}

    path = candidates[0]
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        print(f"[warn] label-map JSON {path} is not a dict; falling back to auto names")
        return {}

    name_to_value = {}
    for name, value in raw.items():
        try:
            name_to_value[str(name)] = int(value)
        except (TypeError, ValueError):
            print(f"[warn] skipping non-integer label value {value!r} for name {name!r} in {path}")
    print(f"loaded {len(name_to_value)} label names from {path.name}")
    return name_to_value


def parse_label_merges(cfg, name_to_value):
    """Validate config ``label_merges`` against the label-map JSON.

    Each entry is ``{name: <merged class name>, labels: [<label names>]}``. The
    referenced names must exist in the label-map JSON (the "consistency" check);
    otherwise the script exits with a clear message. Returns a list of
    ``(name, [member_values], [member_names])``.
    """
    raw = cfg.get("label_merges") or []
    if not isinstance(raw, list):
        sys.exit("'label_merges' in the config must be a list of {name, labels} entries")

    merges = []
    errors = []
    seen_merge_names = set()
    seen_member_names = set()

    for i, group in enumerate(raw):
        where = f"label_merges[{i}]"
        if not isinstance(group, dict):
            errors.append(f"{where}: must be a dict with 'name' and 'labels'")
            continue
        name = group.get("name")
        member_names = group.get("labels")

        if not isinstance(name, str) or not name.strip():
            errors.append(f"{where}: missing a non-empty string 'name'")
            name = None
        elif name == "background":
            errors.append(f"{where}: 'background' is reserved and cannot be a merge name")
        elif name in seen_merge_names:
            errors.append(f"{where}: duplicate merge name '{name}'")
        if name:
            seen_merge_names.add(name)

        if not isinstance(member_names, list) or not member_names:
            errors.append(f"{where}: 'labels' must be a non-empty list of label names")
            member_names = []

        values = []
        for mn in member_names:
            if not isinstance(mn, str):
                errors.append(f"{where}: label {mn!r} is not a string")
                continue
            if mn not in name_to_value:
                errors.append(f"{where}: label '{mn}' not found in the label-map JSON")
                continue
            if mn in seen_member_names:
                errors.append(f"{where}: label '{mn}' is already used in another merge group")
            seen_member_names.add(mn)
            values.append(name_to_value[mn])

        if name and values:
            merges.append((name, sorted(values), member_names))

    if errors:
        sys.exit("label_merges is inconsistent with the label-map JSON:\n  - " + "\n  - ".join(errors))
    return merges


def build_label_mapping(values, value_to_name, merges):
    """Map original label values to consecutive 1..N (0 = background), applying merges.

    ``merges`` is a list of ``(name, [member_values], [member_names])``. Every
    member value present in the data is fused into one class named ``name``;
    unmerged values keep their individual names. Returns (mapping, old_to_name,
    label_names).
    """
    values_set = set(values)
    merged_values = set()
    final_classes = []          # (min_value, name, [values])

    for name, mvals, _ in merges:
        present = [v for v in mvals if v in values_set]
        if not present:
            print(f"[warn] merge '{name}' has no members present in the data; skipped")
            continue
        missing = [v for v in mvals if v not in values_set]
        if missing:
            print(f"[warn] merge '{name}': values {missing} are absent from the data; ignored")
        merged_values.update(present)
        final_classes.append((min(present), name, sorted(present)))

    for v in values:
        if v == 0 or v in merged_values:
            continue
        final_classes.append((v, value_to_name.get(v, f"label_{v}"), [v]))

    final_classes.sort(key=lambda c: c[0])

    mapping = {}          # original value -> new consecutive value
    old_to_name = {}      # original value -> final class name
    label_names = {"background": 0}
    new = 1
    for _, name, mvals in final_classes:
        label_names[name] = new
        for v in mvals:
            mapping[v] = new
            old_to_name[v] = name
        new += 1
    return mapping, old_to_name, label_names


def write_remapped_label(src, dst, mapping):
    img = nib.load(src)
    data = np.asanyarray(img.dataobj)
    out = np.zeros(data.shape, dtype=np.int16)
    for old, new in mapping.items():
        out[data == old] = new
    new_img = nib.Nifti1Image(out, img.affine, img.header.copy())
    new_img.set_data_dtype(np.int16)
    nib.save(new_img, dst)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None,
                        help="path to the YAML config (default: config.yaml next to this script)")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite an existing, non-empty target dataset folder")
    parser.add_argument("--verify", action="store_true",
                        help="run nnU-Net's verify_dataset_integrity on the result")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config) if args.config else script_dir / "config.yaml"
    if not config_path.is_file():
        sys.exit(f"config not found: {config_path}")
    cfg = yaml.safe_load(config_path.read_text())

    image_folder = Path(cfg["image_folder"])
    label_folder = Path(cfg["label_folder"])
    output_raw = Path(cfg["output_raw_folder"])
    dataset_id = int(cfg["dataset_id"])
    dataset_name = cfg["dataset_name"]
    channel_names = cfg.get("channel_names", {0: "DESS"})
    case_id_prefix = cfg.get("case_id_prefix", "dess")
    tol = float(cfg.get("origin_tolerance_mm", 0.5))
    file_ending = cfg.get("file_ending", ".nii.gz")

    # Class names: JSON label map from the label folder first, manual overrides last.
    label_map_name_to_value = load_label_map_json(label_folder, cfg)
    value_to_name = {v: n for n, v in label_map_name_to_value.items()}
    manual_override = {int(k): v for k, v in (cfg.get("label_names") or {}).items()}
    for v, n in manual_override.items():
        value_to_name[v] = n
    name_to_value = {n: v for v, n in value_to_name.items()}

    # Merge groups, validated against the label-map JSON.
    merges = parse_label_merges(cfg, name_to_value)

    image_files = sorted(image_folder.glob(f"*{file_ending}"))
    label_files = sorted(label_folder.glob(f"*{file_ending}"))
    if not image_files:
        sys.exit(f"no images found in {image_folder}")
    if not label_files:
        sys.exit(f"no labels found in {label_folder}")

    print(f"images: {len(image_files)} in {image_folder}")
    print(f"labels: {len(label_files)} in {label_folder}")
    print(f"origin tolerance: {tol} mm")

    image_geoms = [read_geometry(p) for p in image_files]
    label_geoms = [read_geometry(p) for p in label_files]

    matched, excluded_images, excluded_labels = pair_by_origin(image_geoms, label_geoms, tol)

    valid, invalid = [], []
    for ig, lg in matched:
        ok, reason = geometry_is_consistent(ig, lg, tol)
        if ok:
            valid.append((ig, lg))
        else:
            invalid.append((ig, lg, reason))

    valid.sort(key=lambda pair: tuple(pair[0]["origin"]))

    dataset_dir = output_raw / f"Dataset{dataset_id:03d}_{dataset_name}"
    # Only treat the target as already populated when it actually holds data
    # (dataset.json or converted files), not when an earlier run left empty dirs.
    already_populated = (dataset_dir / "dataset.json").exists() or any(
        any((dataset_dir / d).glob(f"*{file_ending}")) for d in ("imagesTr", "labelsTr")
    )
    if already_populated and not args.overwrite:
        sys.exit(f"{dataset_dir} already contains a dataset. Delete it or pass --overwrite.")

    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    # Remap only across the cases we keep, so the label set matches the data.
    label_paths = [lg["path"] for _, lg in valid]
    values, per_case_values = collect_label_values(label_paths)
    mapping, old_to_name, label_names = build_label_mapping(values, value_to_name, merges)

    # Post-merge class distribution: how many kept cases contain each class.
    # Deduplicate to distinct new values per case so a case with several member
    # labels of the same merged class is counted only once.
    class_case_counts = Counter()
    for case_orig in per_case_values:
        for new in {mapping.get(v, 0) for v in case_orig}:
            class_case_counts[new] += 1
    label_distribution = {
        name: {"new_value": new, "num_cases": class_case_counts.get(new, 0)}
        for name, new in label_names.items()
    }

    # Record the merged classes (name + original member values + new value).
    merge_records = []
    for name, mvals, mnames in merges:
        present = [v for v in mvals if v in mapping]
        if present:
            merge_records.append({
                "name": name,
                "labels": mnames,
                "original_values": mvals,
                "new_value": mapping[present[0]],
            })

    case_records = []
    for idx, (ig, lg) in enumerate(valid):
        case_id = f"{case_id_prefix}_{idx:03d}"
        shutil.copy2(ig["path"], images_tr / f"{case_id}_0000{file_ending}")
        write_remapped_label(lg["path"], labels_tr / f"{case_id}{file_ending}", mapping)
        case_records.append({
            "case_id": case_id,
            "original_image": ig["name"],
            "original_label": lg["name"],
            "origin_mm": [round(float(x), 4) for x in ig["origin"]],
        })

    generate_dataset_json(
        output_folder=str(dataset_dir),
        channel_names={str(k): v for k, v in channel_names.items()},
        labels=label_names,
        num_training_cases=len(valid),
        file_ending=file_ending,
        dataset_name=dataset_name,
        overwrite_image_reader_writer=cfg.get("overwrite_image_reader_writer") or None,
        reference="3DReasonKnee / OAI — DESS images with BML labels",
        licence=cfg.get("license", "See OAI / 3DReasonKnee terms of use"),
        converted_by="workspace/convert_dess_to_nnunet.py",
    )

    report = {
        "dataset_dir": str(dataset_dir),
        "num_images_found": len(image_files),
        "num_labels_found": len(label_files),
        "num_cases_kept": len(valid),
        "num_excluded_images": len(excluded_images),
        "num_excluded_labels": len(excluded_labels),
        "num_invalid_pairs": len(invalid),
        "label_mapping": {
            "background": {"new_value": 0, "name": "background"},
            **{str(old): {"new_value": new, "name": old_to_name[old]}
               for old, new in mapping.items()},
        },
        "label_merges": merge_records,
        "label_distribution": label_distribution,
        "excluded_images": [{"file": g["name"], "reason": r} for g, r in excluded_images],
        "excluded_labels": [{"file": g["name"], "reason": r} for g, r in excluded_labels],
        "invalid_pairs": [
            {"image": ig["name"], "label": lg["name"], "reason": r}
            for ig, lg, r in invalid
        ],
        "cases": case_records,
    }
    report_path = script_dir / f"conversion_report_Dataset{dataset_id:03d}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # ---- summary ----------------------------------------------------------- #
    print("\n================ conversion summary ================")
    print(f"dataset        : {dataset_dir}")
    print(f"cases kept     : {len(valid)}")
    print(f"images excluded: {len(excluded_images)}")
    for g, r in excluded_images:
        print(f"    [image] {g['name']} -> {r}")
    print(f"labels excluded: {len(excluded_labels)}")
    for g, r in excluded_labels:
        print(f"    [label] {g['name']} -> {r}")
    print(f"invalid pairs  : {len(invalid)}")
    for ig, lg, r in invalid:
        print(f"    {ig['name']} <-> {lg['name']} -> {r}")
    print("\nlabel remapping (original -> new, name):")
    print("    0 -> 0 background")
    for old, new in mapping.items():
        print(f"    {old} -> {new} {old_to_name[old]}")
    if merge_records:
        print("\nlabel merges:")
        for mr in merge_records:
            print(f"    {mr['name']} (new {mr['new_value']}) <- values {mr['original_values']} {mr['labels']}")
    print("\nclass distribution (num cases containing each class):")
    for name, new in label_names.items():
        print(f"    {name} (new {new}): {label_distribution[name]['num_cases']} cases")
    print(f"\nreport written to: {report_path}")

    if args.verify:
        from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity
        print("\nrunning verify_dataset_integrity ...")
        verify_dataset_integrity(str(dataset_dir))


if __name__ == "__main__":
    main()
