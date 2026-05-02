import pandas as pd


def get_valid_continuous_segments(df: pd.DataFrame, target_frames_count: int) -> list:
    available_indices = df[df["global_is_valid"]]["frame_idx"].tolist()
    available_indices = sorted(available_indices)

    runs = []
    start = 0
    for i in range(1, len(available_indices) + 1):
        if i == len(available_indices) or available_indices[i] != available_indices[i - 1] + 1:
            runs.append((start, i - 1))
            start = i

    available_segments = []

    total = 0
    for s, e in runs:
        if e - s + 1 < target_frames_count:
            continue
        pos = s
        while pos + target_frames_count - 1 <= e:
            start_idx = available_indices[pos]
            end_idx   = available_indices[pos + target_frames_count - 1]

            available_segments.append((start_idx, end_idx))

            total += 1
            pos += target_frames_count

    return available_segments

def check_valid_in_high_fps(df_high: pd.DataFrame, segment_low: tuple) -> bool:
    start_low, end_low = segment_low
    # check if all frames in this segment are valid in high fps
    frame_idx_in_high = [idx for idx in range(start_low * 2, end_low * 2 + 1)]
    for idx in frame_idx_in_high:
        if idx not in df_high[df_high["global_is_valid"]]["frame_idx"].tolist():
            return False
            
    return True 
            


if __name__ == "__main__":
    # FPS 30
    MAIN_INDEX = 1
    FPS = 30
    df_30 = pd.read_csv(f"./data/AnimeFantasyRPG_3_60_preprocessed/{MAIN_INDEX}_fps_{FPS}_merged_frame_index.csv", dtype={"reason_easy": "string", "reason_medium": "string"})
    segments_30 = get_valid_continuous_segments(df_30, target_frames_count=FPS * 2)
    print(f"Valid continuous segments in fps 30 with at least {FPS * 2} valid frames:")
    for start, end in segments_30:
        print(f"Start Frame: {start}, End Frame: {end}")

    # FPS 60
    df_60 = pd.read_csv(f"./data/AnimeFantasyRPG_3_60_preprocessed/{MAIN_INDEX}_fps_{FPS * 2}_merged_frame_index.csv", dtype={"reason_easy": "string", "reason_medium": "string"})
    segments_60 = get_valid_continuous_segments(df_60, target_frames_count=FPS * 4)
    print(f"Valid continuous segments in fps 60 with at least {FPS * 4} valid frames:")
    for start, end in segments_60:
        print(f"Start Frame: {start}, End Frame: {end}")

    # FPS 30 & FPS 60 might have non-overlapping segments
    # find overlapping segments between fps 30 and fps 60 or 
    # find segments in fps 30 then check validity in fps 60

    # check validity of fps 30 segments in fps 60
    valid_segments_in_both = []
    for start_30, end_30 in segments_30:
        # check if all frames in this segment are valid in fps 60
        is_valid_in_60 = check_valid_in_high_fps(df_60, (start_30, end_30))
            
        if is_valid_in_60:
            print(f"Segment valid in both fps 30 and fps 60: Start Frame: {start_30}, End Frame: {end_30}")
        else:
            print(f"Segment NOT valid in fps 60: Start Frame: {start_30}, End Frame: {end_30}")

    
