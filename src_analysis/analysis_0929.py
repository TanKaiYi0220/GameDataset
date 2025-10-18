import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def vis_scatter_chart(name, data_frame, record_name, index_key="FrameIndex", score_key="Score", x_gap=20):
    # visualize all results with matplotlib
    plt.figure(figsize=(10, 6))

    # visualize EPE scores with different modes in one chart
    for mode, group in data_frame.groupby("Mode"):
        # only scatter is enough for line chart smaller
        plt.scatter(group[index_key], group[score_key], marker='o', s=12)
        mean_label = f'Mean EPE [{mode}]= {group[score_key].mean():.4f}'
        plt.axhline(y=group[score_key].mean(), color='g', linestyle='--', label=mean_label)
        median_label = f'Median EPE [{mode}] = {group[score_key].median():.4f}'
        plt.axhline(y=group[score_key].median(), color='r', linestyle='-.', label=median_label)

    plt.title(f"EPE Scores Scatter Chart - {record_name}")
    # x軸間隔 - 10 frames as a gap
    plt.xticks(range(0, data_frame["FrameIndex"].max()+1, x_gap))
    plt.xlabel("Sample Index")
    plt.ylabel("EPE Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(name)
    plt.close()

def mode_rename(mode):
    difficult, _, fps, _ = mode.split("/")
    return f"{difficult}_{fps}"

def get_diff(epe_scores_overall_df, mode_1, mode_2, output_root_path, record_name, x_gap):
    # diff between each frame from two modes
    two_modes = [mode_1, mode_2]
    diff_df = pd.DataFrame()
    diff_df["FrameIndex"] = epe_scores_overall_df["FrameIndex"].unique()
    diff_df[two_modes[0]] = epe_scores_overall_df[epe_scores_overall_df["Mode"] == two_modes[0]]["Score"].values
    diff_df[two_modes[1]] = epe_scores_overall_df[epe_scores_overall_df["Mode"] == two_modes[1]]["Score"].values
    diff_df["DiffScore"] =  diff_df[two_modes[0]] - diff_df[two_modes[1]]
    diff_df["Mode"] = f"{two_modes[0]} vs {two_modes[1]}"

    vis_scatter_chart(
        f"{output_root_path}/epe_scores_diff_chart_{two_modes[0]}_vs_{two_modes[1]}.png", 
        diff_df, f"{record_name} - {two_modes[0]} vs {two_modes[1]}",
        "FrameIndex", "DiffScore",
        x_gap
    )

    print(f"{output_root_path}/epe_scores_diff.csv")
    diff_df.to_csv(f"{output_root_path}/epe_scores_diff.csv")

def main():
    record_name = "AnimeFantasyRPG_2_60"
    fps = "fps_60"
    output_root_path = f"./output/SEARAFT/AnimeFantasyRPG/{record_name}/"
    dataset_mode_path = [
        ## AnimeFantasyRPG_3_60
        # f"0_Easy/0_Easy_0/{fps}/", 
        # f"0_Medium/0_Medium_0/{fps}/", 

        # f"4_Easy/4_Easy_0/{fps}/", 
        # f"4_Medium/4_Medium_0/{fps}/",

        ## AnimeFantasyRPG_2_60
        # f"0_Easy/0_Easy_1/{fps}/", 
        # f"0_Medium/0_Medium_1/{fps}/",

        f"4_Easy/4_Easy_1/{fps}/", 
        f"4_Medium/4_Medium_1/{fps}/",
    ]

    comparison_name = f"{mode_rename(dataset_mode_path[0])}_vs_{mode_rename(dataset_mode_path[1])}"
    analysis_path = f"./analysis_results/0929/{record_name}/{comparison_name}/"
    os.makedirs(analysis_path, exist_ok=True)

    epe_scores_overall_df = pd.DataFrame()

    for mode in dataset_mode_path:
        json_path = f"{output_root_path}/{mode}/results.json"


        with open(json_path, "r") as f:
            data = json.load(f)
            all_results = data["epe_results"]["all"]
            epe_scores_df = pd.DataFrame()
            epe_scores_df["FrameIndex"] = [item["index"] for item in all_results]
            epe_scores_df["Score"] = [item["score"] for item in all_results]
            epe_scores_df["Mode"] = mode_rename(mode)

            print("Json Path:", json_path)
            print("=== Original EPE Statistics ===")
            print("Count (Original) = ", len(epe_scores_df))
            print(f"Mean EPE (Original) = {epe_scores_df['Score'].mean()}")
            print(f"Median EPE (Original) = {epe_scores_df['Score'].median()}")
            print(f"Min  EPE (Original) = {epe_scores_df['Score'].min()}, index = {epe_scores_df['FrameIndex'].idxmin()}")
            print(f"Max  EPE (Original) = {epe_scores_df['Score'].max()}, index = {epe_scores_df['FrameIndex'].idxmax()}")

            # concat different mode results
            epe_scores_overall_df = pd.concat(
                [epe_scores_overall_df, epe_scores_df], 
                ignore_index=True
            )
            epe_scores_overall_df.to_csv(f"{analysis_path}/epe_scores_overall.csv")

    # visualize EPE scores with different modes in each chart
    vis_scatter_chart(
        f"{analysis_path}/epe_scores_chart_overall.png", 
        epe_scores_overall_df, record_name,
        x_gap = 20 if fps == "fps_30" else 40
    )

    for mode in dataset_mode_path:
        mode = mode_rename(mode)
        vis_scatter_chart(
            f"{analysis_path}/epe_scores_chart_{mode}.png", 
            epe_scores_overall_df[epe_scores_overall_df["Mode"] == mode], f"{mode}",
            x_gap = 20 if fps == "fps_30" else 40
        )

    get_diff(
        epe_scores_overall_df, 
        mode_rename(dataset_mode_path[0]), mode_rename(dataset_mode_path[1]), 
        analysis_path, record_name,
        20 if fps == "fps_30" else 40
    )

if __name__ == "__main__":
    main()