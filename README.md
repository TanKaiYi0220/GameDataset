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

## Additional Packages
``` bash
# RIFE
pip install scikit-image    # Successfully installed imageio-2.37.0 scikit-image-0.25.2

# SGM-VFI
pip insall timm==0.9.16     # Successfully installed safetensors-0.6.2 timm-0.9.16

# Sklearn
pip install scikit-learn    # Successfully installed joblib-1.5.2 scikit-learn-1.7.2 threadpoolctl-3.6.0
```