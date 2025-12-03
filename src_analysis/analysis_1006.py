import os
import cv2
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def mode_rename(mode):
    difficult, _, fps, _ = mode.split("/")
    return f"{difficult}_{fps}"

def hit_map_rgb_to_count(hit_map_rgb):
    # Define colors in BGR
    color_0 = [255, 0, 0]   # blue for none
    color_1 = [0, 255, 0]   # green for one
    color_2 = [0, 0, 255]   # red for many

    count_0 = ((hit_map_rgb == color_0).all(axis=2)).sum()
    count_1 = ((hit_map_rgb == color_1).all(axis=2)).sum()
    count_2 = ((hit_map_rgb == color_2).all(axis=2)).sum()

    assert (count_0 + count_1 + count_2 == hit_map_rgb.shape[0] * hit_map_rgb.shape[1])

    return count_0, count_1, count_2    

def get_hit_map_df(dir_path, results_dir, output_path):
    hit_map_df = pd.DataFrame(
        columns=[
            "FrameIdx", 
            "Motion_Many", "Motion_One", "Motion_None",
            "Flow_Many", "Flow_One", "Flow_None"
        ]
    )

    with tqdm(range(len(dir_path))) as pbar:
            for i in pbar:
                dir = dir_path[i]
                hit_map_motion = cv2.imread(f"{results_dir}/{dir}/hit_img_gameMotion_forward.png")
                hit_map_flow = cv2.imread(f"{results_dir}/{dir}/hit_img_opticalFlow_forward.png")

                # convert hit_map rgb to counter 
                # (R = 0 = Many, G = 1 = One, B = 2 = None)
                hit_map_motion_count = hit_map_rgb_to_count(hit_map_motion)
                hit_map_flow_count = hit_map_rgb_to_count(hit_map_flow)

                # append to hit_map_df
                hit_map_df.loc[len(hit_map_df)] = [
                    i,
                    hit_map_motion_count[2],  # many = red
                    hit_map_motion_count[1],  # one = green
                    hit_map_motion_count[0],  # none = blue
                    hit_map_flow_count[2],
                    hit_map_flow_count[1],
                    hit_map_flow_count[0]
                ]

                pbar.set_postfix({"path": f"{dir}"})

    hit_map_df.to_csv(output_path)

    return hit_map_df

def visualize_hit_map_stacked_bar_chart(hit_map_df, mode_label, out_dir):
    # ---------- 畫 Motion ----------
    x = hit_map_df["FrameIdx"].tolist()

    fig_m, ax_m = plt.subplots(figsize=(max(8, len(x)*0.08), 4.5), constrained_layout=True)
    m_none = hit_map_df["Motion_None"].values
    m_one  = hit_map_df["Motion_One"].values
    m_many = hit_map_df["Motion_Many"].values

    bar_width = 0.8
    ax_m.bar(x, m_none, label="None", width=bar_width)
    ax_m.bar(x, m_one, bottom=m_none, label="One", width=bar_width)
    ax_m.bar(x, m_many, bottom=m_none + m_one, label="Many", width=bar_width)

    ax_m.margins(x=0, y=0) # 移除預設 data margin
    ax_m.set_ylabel("Percentage (%)")
    ax_m.set_title(f"{mode_label} - Motion Hit Map (Stacked %)")
    ax_m.legend(loc="upper right", ncol=3, frameon=False)
    motion_png = os.path.join(out_dir, f"{mode_label}_motion_stacked_pct.png")
    fig_m.savefig(motion_png)
    plt.close(fig_m)

def plot_category_heatmap_percent(hit_map_df, mode_label, out_dir, source="Motion", every_n=1, dpi=160):
    """
    將 None/One/Many 三列組成 3 x T 的矩陣做 imshow。
    """
    os.makedirs(out_dir, exist_ok=True)
    df = hit_map_df.iloc[::max(1, every_n)].reset_index(drop=True)

    none = df[f"{source}_None"].to_numpy(dtype=float)
    one  = df[f"{source}_One"].to_numpy(dtype=float)
    many = df[f"{source}_Many"].to_numpy(dtype=float)
    total = none + one + many
    total[total == 0] = 1.0
    mat = np.vstack([none/total*100.0, one/total*100.0, many/total*100.0])

    fig, ax = plt.subplots(figsize=(max(8, mat.shape[1]*0.03), 2.6), constrained_layout=False)
    im = ax.imshow(mat, aspect="auto", origin="lower", vmin=0, vmax=100)
    ax.set_yticks([0,1,2], ["None","One","Many"])
    ax.set_xlabel("FrameIdx")
    ax.set_title(f"{mode_label} - {source} (%) Heatmap")
    ax.margins(x=0)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("%")

    out_path = os.path.join(out_dir, f"{mode_label}_{source}_heatmap_pct.png")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, dpi=dpi)
    plt.close(fig)
    return out_path

def plot_stacked_area_percent(hit_map_df, mode_label, out_dir, source="Motion", every_n=1, dpi=160):
    """
    source: "Motion" 或 "Flow"
    """
    os.makedirs(out_dir, exist_ok=True)
    df = hit_map_df.iloc[::max(1, every_n)].reset_index(drop=True)

    x = df["FrameIdx"].to_numpy()
    none = df[f"{source}_None"].to_numpy(dtype=float)
    one  = df[f"{source}_One"].to_numpy(dtype=float)
    many = df[f"{source}_Many"].to_numpy(dtype=float)

    total = none + one + many
    total[total == 0] = 1.0  # 避免除以 0
    none, one, many = none/total*100.0, one/total*100.0, many/total*100.0

    # fig, ax = plt.subplots(figsize=(max(8, len(x)*0.05), 3.8), constrained_layout=False)
    fig, ax = plt.subplots(constrained_layout=False)
    ax.stackplot(x, none, one, many, labels=["None", "One", "Many"])

    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"{mode_label} - {source} (100% Stacked Area)")
    ax.margins(x=0)
    ax.set_xlim(x.min(), x.max())
    ax.legend(loc="upper right", ncol=3)
    ax.set_xticks(range(0,len(x)+1, 20))
    ax.grid(True)

    out_path = os.path.join(out_dir, f"{mode_label}_{source}_stacked_area_pct.png")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    record_name = "AnimeFantasyRPG_3_60"
    output_root_path = f"./output/SEARAFT/AnimeFantasyRPG/{record_name}/"
    dataset_mode_path = [
        # "0_Easy/0_Easy_0/fps_30/", 
        # "0_Medium/0_Medium_0/fps_30/", 

        "4_Easy/4_Easy_0/fps_30/", 
        "4_Medium/4_Medium_0/fps_30/",
    ]

    analysis_path = f"./analysis_results/1006/{record_name}/"
    os.makedirs(analysis_path, exist_ok=True)


    for mode in dataset_mode_path:
        results_dir = f"{output_root_path}/{mode}/results/"
        frames_dir = sorted(os.listdir(results_dir))
        # frames_dir = frames_dir[:100]

        hit_map_df = get_hit_map_df(frames_dir, results_dir, f"{analysis_path}/{mode_rename(mode)}_hit_map_counter")

        # visualize the percentage of (Motion_One | Motion_Many | Motion_None) by stacked bar chart, also for Flow
        # visualize_hit_map_stacked_bar_chart(hit_map_df, mode_rename(mode), analysis_path)

        # plot_category_heatmap_percent(hit_map_df, mode_rename(mode), analysis_path, "Motion")
        plot_stacked_area_percent(hit_map_df, mode_rename(mode), analysis_path, "Motion")
        

        

            

if __name__ == "__main__":
    main()