# master_thesis_code

Code accompanying the master's thesis on tumor-grade classification from
multi-parametric MRI/PET imaging. Three pipelines at different spatial
granularities share a common data-loading, training and evaluation layer:

- [voxel_wise/](voxel_wise/) — per-voxel features fed to classical ML
  classifiers with an optional hybrid clustering step aggregated to patient-level via majority vote or attention MIL.
- [slice_wise/](slice_wise/) — 2D CNN backbones operating on individual
  slices aggregated to patient-level via majority vote or attention MIL.
- [volume_wise/](volume_wise/) — 3D CNN backbones operating on whole
  volumes.
- [shared/](shared/) — dataset loaders, cross-validation splitting,
  training loop, evaluation, and classifier zoo
  (MLP, logistic regression, decision tree, kNN, SVM, random forest,
  GMM, majority baseline, TabPFN).

Each pipeline is launched with [Hydra](https://hydra.cc/), reading the
`config.yaml` in its own directory.

## Datasets (not included)

The two datasets are not distributed with this repository. They must
be obtained separately and placed in a `Data/` directory at the project
root (the path the pipelines resolve with
`Path(__file__).resolve().parents[3] / "Data"`).

- **Naive40** — cohort from Uniklinikum Tübingen, not publicly available.
  The pipelines expect a single MATLAB file `Naive40_export.mat`
  (see `dataset.naive40.mat_file` in each `config.yaml`).
- **BraTS 2019** — can be downloaded from Kaggle:
  <https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019>.
  Extract it so that the directory `MICCAI_BraTS_2019_Data_Training/`
  (containing `name_mapping.csv` and the per-patient subfolders) lives
  inside `Data/`. The dataset path is configured via
  `dataset.brats.data_dir` in each `config.yaml`.

Expected layout:

```
Data/
├── Naive40_export.mat                 # not publicly available
└── MICCAI_BraTS_2019_Data_Training/   # from Kaggle
    ├── name_mapping.csv
    ├── HGG/
    └── LGG/
```

## Backbones

The slice-wise pipeline uses backbones that are fetched automatically at
runtime (torchvision, HuggingFace Hub, TensorFlow Hub via `medim`/`keras`)
and does not need any manual setup.

The volume-wise pipeline supports two 3D backbones, neither of which is
included in this repository:

### MedicalNet

1. Create the directory `volume_wise/backbones/medicalnet/`.
2. Clone or download the MedicalNet source so that the package
   `volume_wise/backbones/medicalnet/medicalnet/` is importable (the
   model code is loaded via `from .backbones.medicalnet import
   medicalnet` in [volume_wise/model.py](volume_wise/model.py)).
   Source: <https://github.com/Tencent/MedicalNet>.
3. Download the pretrained weights from the MedicalNet release and
   place them under `volume_wise/backbones/medicalnet/pretrain/`
   (e.g. `resnet_10.pth`, `resnet_18.pth`, …).
4. Point `model.volume_cnn.pretrained_path` in
   [volume_wise/config.yaml](volume_wise/config.yaml) at the checkpoint
   you want to use, and set `model.volume_cnn.model_depth` to match
   (10/18/34/50/101/152/200).

### SAM-Med3D

1. Create the directory `volume_wise/backbones/sam_med3d/`.
2. Download `image_encoder3D.py` (the 3D image encoder module) from
   the SAM-Med3D repository
   (<https://github.com/uni-medical/SAM-Med3D>) and place it in that
   directory.
3. Download the pretrained checkpoint `sam_med3d_turbo.pth` from the
   SAM-Med3D releases and place it next to `image_encoder3D.py` (so
   the file lives at `volume_wise/backbones/sam_med3d/sam_med3d_turbo.pth`).
4. Switch the backbone in [volume_wise/config.yaml](volume_wise/config.yaml)
   by setting `model.volume_cnn.backbone: sam_med3d` and pointing
   `model.volume_cnn.pretrained_path` at the checkpoint above.

## Installation

Python dependencies are pinned in [requirements.txt](requirements.txt)
to the versions that were used for the experiments on the cluster. Set
up a fresh environment (e.g. `conda create -n master_thesis python=3.11`
or `python -m venv .venv`) and install with:

```
pip install -r requirements.txt
```

Note that `torch`, `torchvision` and `tensorflow` are pinned to the
Linux/CUDA builds that match the cluster. On a different OS or CUDA
toolchain follow the official installation matrices for those packages
rather than letting pip resolve them blindly.

## Running

Every aspect of a run (dataset, feature columns, backbone, classifier
type and its hyperparameters, cross-validation folds, patient-level
aggregation, training schedule, optional phase-2 fine-tuning, etc.) is
controlled from the pipeline's `config.yaml`. Editing that file (or
overriding values on the command line) is the intended way to configure
an experiment.

Each pipeline is run as a module from the parent directory of
`master_thesis_code/` so that the relative imports of `shared` resolve:

```
python -m master_thesis_code.voxel_wise.main
python -m master_thesis_code.slice_wise.main
python -m master_thesis_code.volume_wise.main
```

Hydra overrides work as usual, e.g.
`python -m master_thesis_code.volume_wise.main dataset.name=brats model.volume_cnn.model_depth=18`.

In addition to the modular `main.py` entry points, the volume- and
slice-wise directories ship simple end-to-end CNN baselines as
standalone scripts:
[slice_wise/2D_CNN.py](slice_wise/2D_CNN.py) trained from
scratch on 2D slices and
[volume_wise/3D_CNN.py](volume_wise/3D_CNN.py) trained from scratch on 3D volumes. They read the same `config.yaml` as
their modular counterparts but do not use any pretrained backbones.

### Hybrid clustering (voxel-wise)

The voxel-wise pipeline has an optional cluster-filtering step
implemented in [voxel_wise/hybrid_clustering/](voxel_wise/hybrid_clustering/).
When enabled, the tumor voxels of each training fold are clustered
(KMeans or GMM) and the most "discriminative" cluster is identified by
the configured `criterion`. The base classifier is then trained only on
voxels assigned to that cluster, while voxels outside the cluster fall
back to a default-class prediction. Toggle and tune it via the
`hybrid_clustering` block in
[voxel_wise/config.yaml](voxel_wise/config.yaml):

```yaml
hybrid_clustering:
  enabled: true           # default false — turn on to use the hybrid path
  method: kmeans          # or 'gmm'
  n_clusters: 3
  find_optimal_k: false   # search k per fold over k_range when true
  k_range: [2, 8]
  criterion: percentage_diff
  min_cluster_voxels: 10
```

The companion script
[voxel_wise/hybrid_clustering/explore_clustering.py](voxel_wise/hybrid_clustering/explore_clustering.py)
runs the clustering sweep separately, e.g. to pick a fixed `n_clusters`
before enabling the hybrid path in the main run.

### Phase-2 fine-tuning (slice- and volume-wise)

Both CNN pipelines support a two-phase training schedule. Phase 1 trains
a classifier head on top of a frozen pretrained backbone (the default
when no `phase2` block is set). Phase 2 resumes from the per-fold
Phase-1 checkpoints and continues training with the last N backbone
stages unfrozen, at a lower learning rate.

To enable it, point `phase2.checkpoint_path` at any per-fold checkpoint
from a previous Phase-1 run (the loader resolves the other folds
automatically) and set how many stages to unfreeze:

```yaml
phase2:
  checkpoint_path: volume_wise/outputs/2026-05-16/11-42-18/0/checkpoint_fold_2.pth
  unfreeze_n_blocks: 2       # 0 = adapter+head only, 1 = layer4,
                             # 2 = layer3+4, -1 = all stages
  learning_rate: 0.0001
  num_epochs: 30
  batch_size: 2
```

The same pattern applies to the slice-wise pipeline via its (currently
commented-out) `phase2` block in
[slice_wise/config.yaml](slice_wise/config.yaml); the field there is
called `unfreeze_n_layers`. Comment the block out (or remove
`checkpoint_path`) to fall back to a fresh Phase-1 run.

## Outputs

Each run is written to a Hydra-managed directory:

- Single run: `<pipeline>/outputs/<YYYY-MM-DD>/<HH-MM-SS>/`
- Multirun (Hydra sweep): `<pipeline>/outputs/multirun/<YYYY-MM-DD>/<HH-MM-SS>/<job_num>/`

Inside each run directory you'll find the resolved Hydra config under
`.hydra/`, the run log, the saved cross-validation metrics, and one
subfolder per classifier type containing per-fold checkpoints
(`checkpoint_fold_*.pth`) and fold-level results.

## Compute environment

The experiments were run on a HPC cluster. Jobs were submitted through
the SLURM workload manager. Deep-learning training and inference were
run on the `gpu-a30` partition with one NVIDIA A30 GPU, 20 CPU cores and
300 GB of host memory per job, with a time budget of 20 hours. The
voxel-wise pipeline, which does not require a GPU, was executed on CPU
nodes with 8 cores and 120 GB of memory.
