import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def vis_line_chart(name, data_frame, upper, lower):
    # visualize all results with matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(data_frame, label="EPE Scores", marker='o', linestyle='None')
    plt.axhline(y=data_frame.mean().values[0], color='g', linestyle='--', label='Mean EPE (Filtered)')
    plt.axhline(y=upper.values[0], color='r', linestyle='--', label='Upper Bound (IQR)')
    plt.axhline(y=lower.values[0], color='b', linestyle='--', label='Lower Bound (IQR)')
    plt.title('EPE Scores with IQR Filtering')
    plt.xlabel('Sample Index')
    plt.ylabel('EPE Score')
    plt.legend()
    plt.grid()
    plt.savefig(name)
    plt.close()

def main():
    output_root_path = "output/SEARAFT/AnimeFantasyRPG/"
    dataset_mode_path = [
        "0_Easy/0_Easy_0/fps_30/", 
        "0_Medium/0_Medium_0/fps_30/", 
        "0_Difficult/0_Difficult_0/fps_30/"
    ]

    for mode in dataset_mode_path:
        json_path = f"{output_root_path}/{mode}/exp_results.json"

        with open(json_path, "r") as f:
            data = json.load(f)
            all_results = data["epe_results"]["all"]
            epe_scores_df = pd.DataFrame([item["score"] for item in all_results])


            # filter outliers with IQR
            Q1 = epe_scores_df.quantile(0.25)
            Q3 = epe_scores_df.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            filtered_scores_df = epe_scores_df[(epe_scores_df >= lower) & (epe_scores_df <= upper)].dropna()
            mean_epe = filtered_scores_df.mean().values[0]

            vis_line_chart(
                f"{output_root_path}/{mode}/epe_scores_chart.png", 
                epe_scores_df, upper, lower
            )

            print("Json Path:", json_path)
            print("=== Original EPE Statistics ===")
            print("Count (Original) = ", len(epe_scores_df))
            print(f"Mean EPE (Original) = {epe_scores_df.mean().values[0]}")
            print(f"Min  EPE (Original) = {epe_scores_df.min().values[0]}, index = {epe_scores_df.idxmin().values[0]}")
            print(f"Max  EPE (Original) = {epe_scores_df.max().values[0]}, index = {epe_scores_df.idxmax().values[0]}")

            print("=== Filtered EPE Statistics ===")
            print("Count (Filtered) = ", len(filtered_scores_df))
            print(f"Mean EPE (Filtered) = {mean_epe}")
            print(f"Min  EPE (Filtered) = {filtered_scores_df.min().values[0]}, index = {filtered_scores_df.idxmin().values[0]}")
            print(f"Max  EPE (Filtered) = {filtered_scores_df.max().values[0]}, index = {filtered_scores_df.idxmax().values[0]}")

            print("=== Over Upper EPE Statistics ===")
            over_upper_df = epe_scores_df[epe_scores_df > upper].dropna()
            print("Count (Over Upper) = ", len(over_upper_df))
            print(f"Mean EPE (Over Upper) = {over_upper_df.mean().values[0]}")
            print(f"Min  EPE (Over Upper) = {over_upper_df.min().values[0]}, index = {over_upper_df.idxmin().values[0]}")
            print(f"Max  EPE (Over Upper) = {over_upper_df.max().values[0]}, index = {over_upper_df.idxmax().values[0]}")
        print()

if __name__ == "__main__":
    main()