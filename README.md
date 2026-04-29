# GameDataset
## Commands
``` bash
python main.py --input_folder "./datasets/Fantasy_RPG/FRPG_0_0_0/fps_60/" --output_folder "./datasets/Fantasy_RPG/FRPG_0_0_0/fps_60_png/" --file_type "colorNoScreenUI"
```


``` bash
python demo_searaft.py --cfg "./models/SEARAFT/config/eval/spring-M.json" --model "./models/SEARAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth"
```

``` bash
python inference_searaft.py --cfg "./models/SEARAFT/config/eval/spring-M.json" --model "./models/SEARAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth"
```


``` bash
python3 inference_rife.py --exp=1 --model ./models/RIFE/train_log
```

``` bash
python src_analysis/analysis_0929.py
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
docker run --rm --gpus all -v /path/to/datasets:/workspace/datasets -v "%cd%":/app gfi:docker-version python train_ifrnet.py
```

If you need to use a custom local config file, create `config/user_config.json` and do not commit it.

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