import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from src.warp_module import Warping
from src.utils import find_max_index_in_dir, save_img, EXR_to_PNG


import sys
sys.path.append('models/RIFE')

import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

@torch.no_grad()
def inference(name, args, model, warping_module, image1, image2):
    n, c, h, w = image1.shape
    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64
    padding = (0, pw - w, 0, ph - h)
    image1 = F.pad(image1, padding)
    image2 = F.pad(image2, padding)

    if args.ratio:
        if model.version >= 3.9:
            img_list = [image1, model.inference(image1, image2, args.ratio), image2]
        else:
            img0_ratio = 0.0
            img1_ratio = 1.0
            if args.ratio <= img0_ratio + args.rthreshold / 2:
                middle = image1
            elif args.ratio >= img1_ratio - args.rthreshold / 2:
                middle = image2
            else:
                tmp_img0 = image1
                tmp_img1 = image2
                for inference_cycle in range(args.rmaxcycles):
                    middle = model.inference(tmp_img0, tmp_img1)
                    middle_ratio = ( img0_ratio + img1_ratio ) / 2
                    if args.ratio - (args.rthreshold / 2) <= middle_ratio <= args.ratio + (args.rthreshold / 2):
                        break
                    if args.ratio > middle_ratio:
                        tmp_img0 = middle
                        img0_ratio = middle_ratio
                    else:
                        tmp_img1 = middle
                        img1_ratio = middle_ratio
            img_list.append(middle)
            img_list.append(image2)
    else:
        if model.version >= 3.9:
            img_list = [image1]
            n = 2 ** args.exp
            for i in range(n-1):
                img_list.append(model.inference(image1, image2, (i+1) * 1. / n))
            img_list.append(image2)
        else:
            img_list = [image1, image2]
            for i in range(args.exp):
                tmp = []
                for j in range(len(img_list) - 1):
                    mid = model.inference(img_list[j], img_list[j + 1])
                    tmp.append(img_list[j])
                    tmp.append(mid)
                tmp.append(image2)
                img_list = tmp

    output_path = f"output/{name}/"
    os.makedirs(output_path, exist_ok=True)

    for i in range(len(img_list)):
        cv2.imwrite(f'{output_path}img{i}.png', (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

    return 0.0




def FRPG_loader(name, device, model, warping_module, dataset_dir_path, args):
    max_index = find_max_index_in_dir(dataset_dir_path)
    print(max_index)

    epe_scores = []

    with tqdm(range(max_index - 1)) as pbar:
        for i in pbar:
            image_1_path = f"{dataset_dir_path}colorNoScreenUI_{i}.exr"
            image_2_path = f"{dataset_dir_path}colorNoScreenUI_{i + 1}.exr"
            image_1 = EXR_to_PNG(image_1_path)
            image_2 = EXR_to_PNG(image_2_path)

            save_img(f"{dataset_dir_path}colorNoScreenUI_{i}.png", image_1)
            save_img(f"{dataset_dir_path}colorNoScreenUI_{i + 1}.png", image_2)

            image_1 = cv2.imread(f"{dataset_dir_path}colorNoScreenUI_{i}.png", cv2.IMREAD_UNCHANGED)
            image_2 = cv2.imread(f"{dataset_dir_path}colorNoScreenUI_{i + 1}.png", cv2.IMREAD_UNCHANGED)

            image_1 = (torch.tensor(image_1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
            image_2 = (torch.tensor(image_2.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

            file_name = f"{name}/results/frame_{i:04d}_to_{i+1:04d}/"

            epe = inference(file_name, args, model, warping_module, image_1, image_2)
            pbar.set_postfix({"EPE": f"{epe:.4f}"})

            epe_scores.append(epe)

    # Prepare JSON result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    parser.add_argument('--exp', default=4, type=int)
    parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
    parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')
    parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
    parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')

    args = parser.parse_args()

    from models.RIFE.train_log.RIFE_HDv3 import Model
    model = Model()
    model.load_model(args.modelDir, -1)
    print("Loaded v3.x HD model.")
    model.eval()
    model.device()

    warping_module = Warping(warp_mode="nearest")

    # template for loading FRPG images
    dataset_root_path = "/datasets/VFI/datasets/AnimeFantasyRPG/"
    dataset_mode_path = [
        "0_Easy/0_Easy_0/fps_30/", 
        # "0_Medium/0_Medium_0/fps_30/", 
        # "0_Difficult/0_Difficult_0/fps_30/"
    ]

    for mode in dataset_mode_path:
        dataset_dir_path = os.path.join(dataset_root_path, mode)
        FRPG_loader(f"RIFE/AnimeFantasyRPG/{mode}", device, model, warping_module, dataset_dir_path, args)




if __name__ == "__main__":
    main()