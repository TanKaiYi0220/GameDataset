from metrics.VFIPS import calc_vfips, calc_vfips_mp4


if __name__ == "__main__":
    print("Evaluating VFIPS...")
    # score = calc_vfips("path/to/dis_dir", "path/to/ref_dir")

    score = calc_vfips_mp4(
        pred_mp4="/home/kevin/Desktop/VFI/GFI/videos/IFRNet_FineTuning_Val_2_30_AnimeFantasyRPG_3_60_3_Difficult/3_Difficult_2/fps_60_vfi60_pred.mp4",
        gt_mp4="/home/kevin/Desktop/VFI/GFI/videos/IFRNet_FineTuning_Val_2_30_AnimeFantasyRPG_3_60_3_Difficult/3_Difficult_2/fps_60_vfi60_gt.mp4",
        ckpt_path="metrics/VFIPS/checkpoints/VFIPS.pytorch",
        clip_len=12,
        stride=12,          # 跟你原本一致；想更密就改 1
    )
    print(f"VFIPS Score: {score}")