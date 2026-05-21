# Gaussian Adversarial Attack on the Sybil Classifier

Adversarial PGD attack on MedGS / 3DGS Gaussian parameters using gradients
from the Sybil lung-cancer risk classifier.

## Structure

```
.
├── attack/                       adversarial attack pipeline
│   ├── main.py                   entry point
│   ├── cli.py                    command-line arguments
│   ├── constants.py              Sybil / MedGS constants
│   ├── medgs_bootstrap.py        finds MedGS on sys.path
│   ├── ply_io.py                 raw PLY reading + SH-degree detection
│   ├── geometry.py               camera matrices, 3D covariances, projection
│   ├── camera_loader.py          matches cameras.json with mask files
│   ├── mask_filtering.py         selects Gaussians lying inside masks
│   ├── attribute_mapping.py      PLY fields <-> GaussianModel tensor slices
│   ├── classifier_loader.py      Sybil ensemble (frozen weights)
│   ├── volume_rendering.py       renders slices into a Sybil-shaped volume
│   ├── pgd_attack.py             PGD loop on Gaussian attributes
│   └── scene_setup.py            MedGS Scene initialization
│
└── analysis/                     post-attack analysis
    ├── image_diff.py             per-pixel absolute diffs between two folders
    ├── component_analysis.py     connected components inside the lung masks
    └── overlay_circles.py        draws circles on originals from centroid CSV
```

## End-to-end pipeline

1. Train MedGS on a CT volume.
2. `python -m attack.main ...` runs the adversarial attack and writes an attacked PLY.
3. Render the attacked Gaussians with MedGS to get adversarial slices.
4. `python -m analysis.image_diff` compares attacked vs. clean renders.
5. `python -m analysis.component_analysis` extracts perturbed regions inside the lungs.
6. `python -m analysis.overlay_circles` annotates the originals with the detections.

## Attack example

```bash
python -m attack.main \
  --model_path output/1 \
  --source_path data/1 \
  --masks <patient_id>/mask \
  --output_ply attacked_clf.ply \
  --sigma 0.5 --remove_top_pct 10.0 \
  --props f_dc_0 f_dc_1 f_dc_2 \
  --eps 0.5 --steps 10 \
  --attack_mode untargeted \
  --classifier_config_dir /abs/path/to/SybilInference/configs \
  --classifier_config_name nlst_sybil_ensemble_inference \
  --target_year 5
```

## Key implementation notes

- The attack uses **raw logits** (`out["logit"]`), not `forward_all_years()` —
  sigmoid -> calibrate -> clip would zero the gradient in saturated regions.
- Sybil preprocessing **skips uint8 quantization** (zero-gradient op).
- Sybil preprocessing **skips spacing-based resampling** (rendered slices have
  no DICOM spacing); a trilinear spatial resize is used instead.
- The classifier is frozen (`requires_grad_(False)`), but gradient still flows
  through it back into the Gaussian parameters.
- Only a subset of the ensemble is loaded into RAM by default
  (`--n_ensemble_load 1`) to keep memory low. The Sybil ensemble has 5
  members of ~130 MB each.
