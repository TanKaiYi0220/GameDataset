import os
import sys
sys.path.append('/datasets/VFI/offline_dataset/')
from datasets.dataset_config import DATASET_CONFIGS, STAIR_DATASET_CONFIG, iter_dataset_configs
from datasets.dataset_loader import FlowEstimationDataset
from src.utils import show_images_switchable, flow_to_image
from src.gameData_loader import load_backward_velocity

import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

# ========= K-Means Clustering =========
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT_DIR = "./datasets/data/"
OUTPUT_DIR = "./output/IFRNet/"
ANALYSIS_DIR = f"./analysis_results/1208_DistanceIndexing/"
DATASET = DATASET_CONFIGS


def cosine_project_ratio(array1, array2):
    # calculate the dot product of each pair of vectors in the two arrays
    array_inner = array1[..., 0] * array2[..., 0] + array1[..., 1] * array2[..., 1]
    # array1_mag = np.linalg.norm(array1, axis=1)
    array2_mag = np.linalg.norm(array2, axis=-1)
    array_cos_sim = array_inner / (array2_mag ** 2)

    return array_cos_sim

def assign_quantile_group(value, q):
    if value <= q[0.25]:
        return "Q1"
    elif value <= q[0.50]:
        return "Q2"
    elif value <= q[0.75]:
        return "Q3"
    else:
        return "Q4"



if __name__ == "__main__":
    for cfg in iter_dataset_configs(DATASET):
        if cfg.fps != 60:
            continue

        if cfg.difficulty != "Easy":
            continue

    
        df = pd.read_csv(f"{ROOT_DIR}/{cfg.record_name}_preprocessed/{cfg.mode_index}_raw_sequence_frame_index.csv")
        
        dataset_fps30 = FlowEstimationDataset(
            df=df,
            root_dir=DATASET["root_dir"],
            record=cfg.record,
            mode=cfg.mode_path,
            input_fps=30,
        )

        dataset_fps60 = FlowEstimationDataset(
            df=df,
            root_dir=DATASET["root_dir"],
            record=cfg.record,
            mode=cfg.mode_path,
            input_fps=60,
        )

        result_df_path = f"{OUTPUT_DIR}/{cfg.record}/{cfg.mode_name}_evaluation_results.csv"
        print(result_df_path)
        df = pd.read_csv(result_df_path)

        print(len(dataset_fps30), len(dataset_fps60))

        rows = []

        with tqdm(range(len(dataset_fps30))) as pbar:
            for i in pbar:
                # fps30
                sample_fps30 = dataset_fps30[i]
                input_fps30 = sample_fps30["input"]
                gt_fps30 = sample_fps30["ground_truth"]
                color_fps30_0_path = input_fps30["colorNoScreenUI"][0]
                color_fps30_2_path = input_fps30["colorNoScreenUI"][1]
                backwardVel_fps30_2_0_path = gt_fps30["backwardVel_Depth"]

                # fps60
                sample_fps60 = dataset_fps60[2 * i]
                input_fps60 = sample_fps60["input"]
                gt_fps60 = sample_fps60["ground_truth"]
                color_fps60_0_path = input_fps60["colorNoScreenUI"][0]
                color_fps60_1_path = input_fps60["colorNoScreenUI"][1]
                backwardVel_fps60_1_0_path = gt_fps60["backwardVel_Depth"]

                if i == 0:
                    print(backwardVel_fps30_2_0_path, backwardVel_fps60_1_0_path)


                # load backwardVel
                img0_np = cv2.imread(color_fps30_0_path)
                img1_np = cv2.imread(color_fps60_1_path)
                img2_np = cv2.imread(color_fps30_2_path)

                backwardVel_2_0, _ = load_backward_velocity(backwardVel_fps30_2_0_path)
                backwardVel_1_0, _ = load_backward_velocity(backwardVel_fps60_1_0_path)
                backwardVel_2_0 = backwardVel_2_0[0].data.permute(1, 2, 0).cpu().numpy()
                backwardVel_1_0 = backwardVel_1_0[0].data.permute(1, 2, 0).cpu().numpy()

                dis_index = cosine_project_ratio(backwardVel_1_0, backwardVel_2_0)
                
                if df.at[i, "valid"]:
                    rows.append({
                        "frame index": i,
                        "distance index (mean)": np.mean(dis_index),
                        "distance index (median)": np.median(dis_index),
                        "psnr": df.at[i, "psnr"]
                    })
                # show_images_switchable(
                #     [img0_np, img1_np, img2_np, dis_index],
                #     [f"img0 {i}", f"img1 {i}", f"img2 {i}", f"distance index {df.at[i, 'psnr']}"]
                # )
        
        distance_index_df = pd.DataFrame(rows)
        os.makedirs(f"{ANALYSIS_DIR}/{cfg.record_name}/", exist_ok=True)
        distance_index_df.to_csv(f"{ANALYSIS_DIR}/{cfg.record_name}/{cfg.mode_index}_distance_index.csv", index=False)

        
        # compute quantiles for mean and median
        quantiles_mean = distance_index_df["distance index (mean)"].quantile([0.25, 0.5, 0.75])
        quantiles_median = distance_index_df["distance index (median)"].quantile([0.25, 0.5, 0.75])

        distance_index_df["mean_group"] = distance_index_df["distance index (mean)"].apply(lambda x: assign_quantile_group(x, quantiles_mean))
        distance_index_df["median_group"] = distance_index_df["distance index (median)"].apply(lambda x: assign_quantile_group(x, quantiles_median))

        results = []

        for g in ["Q1", "Q2", "Q3", "Q4"]:
            mean_subset = distance_index_df[distance_index_df["mean_group"] == g]
            median_subset = distance_index_df[distance_index_df["median_group"] == g]

            results.append({
                "group": g,
                "mean↔psnr corr": mean_subset["distance index (mean)"].corr(mean_subset["psnr"]),
                "median↔psnr corr": median_subset["distance index (median)"].corr(median_subset["psnr"]),
                "mean size": len(mean_subset),
                "median size": len(median_subset),
            })

        corr_df = pd.DataFrame(results)

        os.makedirs(f"{ANALYSIS_DIR}/{cfg.record_name}/", exist_ok=True)
        corr_df.to_csv(f"{ANALYSIS_DIR}/{cfg.record_name}/{cfg.mode_index}_distance_index_corr.csv", index=False)
        print(corr_df)


        # 取出要做 clustering 的 features
        feature_cols = ["distance index (mean)", "distance index (median)", "psnr"]
        features = distance_index_df[feature_cols].dropna()

        # 標準化，避免 psnr 尺度遠大於 mean/median
        # 設定 K（可自行調整）
        for K in range(2, 6):
            scaler = StandardScaler()
            X = scaler.fit_transform(features.values)
            kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
            labels = kmeans.fit_predict(X)

            # 把 cluster label 寫回原本的 dataframe（用 index 對齊）
            distance_index_df.loc[features.index, "cluster"] = labels

            # 反標準化 cluster center，方便解讀每一群的大致 mean/median/psnr
            centers_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)

            cluster_summary = pd.DataFrame(
                centers_original_scale,
                columns=feature_cols
            )
            cluster_summary["cluster"] = range(K)
            cluster_summary["cluster_counter"] = cluster_summary["cluster"].apply(lambda x: distance_index_df['cluster'].value_counts()[x])

            # 存 clustering 結果
            distance_index_df.to_csv(
                f"{ANALYSIS_DIR}/{cfg.record_name}/{cfg.mode_index}_distance_index_with_clusters_{K}.csv",
                index=False
            )
            cluster_summary.to_csv(
                f"{ANALYSIS_DIR}/{cfg.record_name}/{cfg.mode_index}_distance_index_cluster_centers_{K}.csv",
                index=False
            )

            print("K-Means cluster centers (original scale):")
            print(cluster_summary)
            print(distance_index_df['cluster'].value_counts())