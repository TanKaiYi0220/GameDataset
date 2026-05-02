# GameDataset
## Commands
``` bash
python -m gamedataset.training.ifrnet
```

``` bash
python -m gamedataset.training.ifrnet_residual
```

``` bash
python -m gamedataset.inference.ifrnet
```

``` bash
python -m gamedataset.inference.ifrnet_residual
```

``` bash
python -m gamedataset.inference.results_to_video --help
```

``` bash
python -m gamedataset.analysis.y2025.analysis_0929
```

## Docker / Multi-user configuration
1. Copy `config/user_config.example.json` to `config/user_config.json`.
2. Edit `dataset_root`, `output_root`, and other paths to match your local or container environment.
3. Build the Docker image:
```bash
docker build -t gfi:docker-version .
```
4. Run from repo root with mounted data and GPU support:
```bash
docker run --rm --gpus all -v /path/to/datasets:/workspace/datasets -v "%cd%":/app gfi:docker-version python -m gamedataset.training.ifrnet
```

If you need to use a custom local config file, create `config/user_config.json` and do not commit it.

## Resume Output Directory Automation
When you do not pass `--output_dir`, training now creates a fresh output directory automatically.

- Fresh training: uses the model default output directory name and appends a numeric suffix if needed.
- Resume training: if `--resume_path` points to an existing checkpoint, the new output directory is created automatically as a sibling of the source run, so the original run is preserved.

Example:
```bash
python -m gamedataset.training.ifrnet --resume_path ./output/IFRNet_FineTuning_Resume_0416/checkpoints/best.pth
```

This will resume from the checkpoint but write new checkpoints, logs, and CSV outputs into a new directory derived from the source run. You only need to pass `--output_dir` when you want a specific custom path.

## VFI Triplet Selection
During dataset preprocessing, the generated `*_raw_sequence_frame_index.csv` now keeps only even-start triplets for VFI:

- `(0, 1, 2)`
- `(2, 3, 4)`
- `(4, 5, 6)`

Odd-start triplets such as `(1, 2, 3)` are skipped before the dataloader stage, so the training samples stay aligned with the intended `fps_30` motion pairs.

## Additional Packages
``` bash
pip install pandas

# RIFE
pip install scikit-image    # Successfully installed imageio-2.37.0 scikit-image-0.25.2

# SGM-VFI
pip install timm==0.9.16     # Successfully installed safetensors-0.6.2 timm-0.9.16

# Sklearn
pip install scikit-learn    # Successfully installed joblib-1.5.2 scikit-learn-1.7.2 threadpoolctl-3.6.0
```
