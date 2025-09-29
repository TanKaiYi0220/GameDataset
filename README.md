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

## Additional Packages
``` bash
pip install scikit-image # Successfully installed imageio-2.37.0 scikit-image-0.25.2
```