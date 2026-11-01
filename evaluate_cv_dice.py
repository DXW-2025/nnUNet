"""
Cross-validation evaluation for a trained nnU-Net model when --val was NOT used
(or to evaluate a still-running / interrupted training).

For each requested fold, predicts that fold's held-out validation cases with the
fold's own weights and computes per-class Dice against the ground-truth labels.

Usage:
    python evaluate_cv_dice.py [dataset_name] [checkpoint_name] [folds]

    dataset_name     defaults to 'Dataset902_KneeDESS_BML'
    checkpoint_name  defaults to 'checkpoint_latest.safetensors'
                     (e.g. 'checkpoint_best.safetensors')
    folds            comma-separated, defaults to '0,1,2,3'

Reads nnUNet_raw / nnUNet_preprocessed / nnUNet_results from the environment.
"""
import os
import sys

import numpy as np

from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json, maybe_mkdir_p

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder

TRAINER = 'nnUNetTrainer'
PLANS = 'nnUNetPlans'
CONFIG = '2d'


def main(dataset='Dataset902_KneeDESS_BML', checkpoint_name='checkpoint_latest.safetensors',
         folds=(0, 1, 2, 3)):
    raw = os.environ['nnUNet_raw']
    pre = os.environ['nnUNet_preprocessed']
    res = os.environ['nnUNet_results']

    model_folder = join(res, dataset, f'{TRAINER}__{PLANS}__{CONFIG}')
    images_tr = join(raw, dataset, 'imagesTr')
    labels_tr = join(raw, dataset, 'labelsTr')

    splits = load_json(join(pre, dataset, 'splits_final.json'))
    dataset_json = load_json(join(raw, dataset, 'dataset.json'))
    id_to_name = {v: k for k, v in dataset_json['labels'].items()}

    per_fold_mean = {}
    per_fold_fg = {}
    per_case = {}

    for fold in folds:
        val_cases = splits[fold]['val']
        input_files = [[join(images_tr, c + '_0000.nii.gz')] for c in val_cases]
        out_folder = join(model_folder, f'fold_{fold}', 'validation_manual')
        maybe_mkdir_p(out_folder)

        print(f'\n=== fold {fold}: predicting {len(val_cases)} val cases -> {out_folder} ===', flush=True)

        predictor = nnUNetPredictor(verbose=False, allow_tqdm=True)
        predictor.initialize_from_trained_model_folder(
            model_folder, use_folds=(fold,), checkpoint_name=checkpoint_name
        )
        predictor.predict_from_files(input_files, out_folder, save_probabilities=False, overwrite=True)

        metrics = compute_metrics_on_folder(
            labels_tr, out_folder, join(out_folder, 'summary.json'),
            predictor.plans_manager.image_reader_writer_class(),
            predictor.dataset_json['file_ending'],
            predictor.label_manager.foreground_regions if predictor.label_manager.has_regions
            else predictor.label_manager.foreground_labels,
            predictor.label_manager.ignore_label,
            chill=True,
        )

        per_fold_mean[fold] = metrics['mean']
        per_fold_fg[fold] = metrics['foreground_mean']
        per_case[fold] = {c['reference_file']: c['metrics'] for c in metrics['metric_per_case']}
        print(f'fold {fold}: foreground Dice = {metrics["foreground_mean"]["Dice"]:.4f}', flush=True)

    label_ids = [lid for lid in per_fold_mean[folds[0]].keys() if lid != 0]
    header = f'{"label":<24}' + ''.join(f'{f"fold {f}":>9}' for f in folds) + f'{"mean":>9} {"std":>8}'
    print('\n================ per-class Dice (mean +/- std across folds) ================')
    print(header)
    print('-' * len(header))
    for lid in label_ids:
        name = id_to_name.get(lid, str(lid))
        vals = [per_fold_mean[f][lid]['Dice'] for f in folds]
        cells = ''.join(f'{v * 100:>8.2f}%' if not np.isnan(v) else f'{"nan":>9}' for v in vals)
        print(f'{name:<24}{cells}{np.nanmean(vals) * 100:>8.2f}%{np.nanstd(vals) * 100:>7.2f}%')

    fg = [per_fold_fg[f]['Dice'] for f in folds]
    print('-' * len(header))
    print(f'{"foreground_mean":<24}{"".join(f"{v*100:>8.2f}%" for v in fg)}{np.nanmean(fg)*100:>8.2f}%{np.nanstd(fg)*100:>7.2f}%')

    save_json({'per_fold_mean': per_fold_mean, 'per_fold_foreground_mean': per_fold_fg,
               'per_case': per_case, 'checkpoint': checkpoint_name},
              join(model_folder, 'crossval_summary_manual.json'))


if __name__ == '__main__':
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'Dataset902_KneeDESS_BML'
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else 'checkpoint_latest.safetensors'
    folds = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else (0, 1, 2, 3)
    main(dataset, checkpoint, folds)
