import pandas as pd
import re

try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)


def _summarize_datetime_index(name: str, idx: pd.DatetimeIndex, preview_count: int = 5):
    print(f"{name} count: {len(idx)}")

    if len(idx) == 0:
        return

    head_vals = list(idx[:preview_count])
    tail_vals = list(idx[-preview_count:])

    print(f"{name} first {min(preview_count, len(idx))}: {head_vals}")

    if len(idx) > preview_count:
        print(f"{name} last {min(preview_count, len(idx))}: {tail_vals}")


def infer_dominant_granularity(df: pd.DataFrame):
    deltas = df.index.to_series().diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]

    if deltas.empty:
        return None

    counts = deltas.value_counts().sort_values(ascending=False)
    dominant = counts.index[0]

    print("Detected source granularities (count by interval):")
    for delta, count in counts.items():
        print(f"- {delta}: {count}")

    print(f"Dominant source granularity: {dominant}")

    return dominant

def signature(value):
    if pd.isna(value):
        return "missing"

    s = str(value).strip()
    s = re.sub(r"\d+", "digits", s)
    return s


def digit_ranges(series):
    digit_groups = []

    for value in series.dropna():
        s = str(value).strip()
        groups = re.findall(r"\d+", s)

        for i, g in enumerate(groups):
            if len(digit_groups) <= i:
                digit_groups.append([])

            digit_groups[i].append(int(g))

    ranges = {}

    for i, values in enumerate(digit_groups, start=1):
        ranges[f"digits_{i}_min"] = min(values)
        ranges[f"digits_{i}_max"] = max(values)

    return ranges


def find_date_stuctural_differences(df, date_column_name):
    temp = df[[date_column_name]].copy()
    temp["_signature"] = temp[date_column_name].apply(signature)

    blocks = []
    current_sig = None
    start_row = None
    previous_row = None

    for row, sig in zip(temp.index, temp["_signature"]):
        if sig != current_sig:
            if current_sig is not None:
                block_values = df.loc[start_row:previous_row, date_column_name]
                ranges = digit_ranges(block_values)

                blocks.append({
                    "from_row": start_row,
                    "to_row": previous_row,
                    "structure": current_sig,
                    **ranges
                })

            current_sig = sig
            start_row = row

        previous_row = row

    if current_sig is not None:
        block_values = df.loc[start_row:previous_row, date_column_name]
        ranges = digit_ranges(block_values)

        blocks.append({
            "from_row": start_row,
            "to_row": previous_row,
            "structure": current_sig,
            **ranges
        })

    structure_report = pd.DataFrame(blocks)

    print("Date report:")
    display(structure_report)

    return structure_report


def report_granularity_blocks(df):
    deltas = df.index.to_series().diff().dropna()

    if deltas.empty:
        print("Not enough rows to detect granularity.")
        return pd.DataFrame()

    blocks = []
    current_delta = None
    block_start = df.index[0]
    previous_time = df.index[0]

    for current_time, delta in deltas.items():
        if current_delta is None:
            current_delta = delta

        elif delta != current_delta:
            blocks.append({
                "granularity": current_delta,
                "from": block_start,
                "to": previous_time
            })

            block_start = previous_time
            current_delta = delta

        previous_time = current_time

    blocks.append({
        "granularity": current_delta,
        "from": block_start,
        "to": df.index[-1]
    })

    report = pd.DataFrame(blocks)

    print("Granularity report:")
    display(report)

    return report


def handle_varying_granularities(df, output_granularity, keep_cols_info):
    target_delta = pd.to_timedelta(output_granularity)
    target_freq = output_granularity

    deltas = df.index.to_series().diff().dropna()

    if deltas.empty:
        return df

    unique_deltas = sorted(deltas.unique())

    print("Unique granularities detected:")
    for d in unique_deltas:
        print(f"- {d}")

    # If already exactly one granularity and it matches target, do nothing
    if len(unique_deltas) == 1 and unique_deltas[0] == target_delta:
        print("Already at desired granularity. No resampling applied.")
        return df

    report_granularity_blocks(df)

    # This handles multiple numeric columns correctly.
    # Downsampling: averages each column.
    # Upsampling: creates target timestamps and interpolates each column.
    if min(unique_deltas) < target_delta:
        print(f"Downsampling at least part of data to {target_freq}")

    if max(unique_deltas) > target_delta:
        print(f"Upsampling at least part of data to {target_freq}")

    resampled_columns = []

    for col in df.columns:
        # Get config for this column
        config = keep_cols_info.get(col)
        method = config["method"]
        
        
        if method == "mean":
            s = df[col].resample(target_freq).mean()

        elif method == "sum":
            s = df[col].resample(target_freq).sum()

        elif method == "first":
            s = df[col].resample(target_freq).first()

        elif method == "last":
            s = df[col].resample(target_freq).last()

        elif method == "ffill":
            s = df[col].resample(target_freq).ffill()

        elif method == "interpolate":
            s = df[col].resample(target_freq).mean()
            s = s.interpolate(method="linear")

        else:
            raise ValueError(f"Unknown resample method for {col}: {method}")

        resampled_columns.append(s)

    df = pd.concat(resampled_columns, axis=1)

    return df


def handle_timeseries_dataframe(
    df: pd.DataFrame,
    date_column_name: str,
    raw_date_format: str,
    expected_first_timestamp: pd.Timestamp,
    expected_last_timestamp: pd.Timestamp,
    df_keep_columns: list,
    keep_cols_info: dict,
    filter_first_timestamp: pd.Timestamp,
    filter_last_timestamp: pd.Timestamp,
    output_granularity: str
):
    # Remove unwanted columns
    keep_cols = [date_column_name] + [
        col for col in df_keep_columns if col != date_column_name
    ]

    df = df[keep_cols].copy()

    # Check date string structures before converting to datetime
    find_date_stuctural_differences(df, date_column_name)

    # Convert date column
    df[date_column_name] = pd.to_datetime(
        df[date_column_name],
        format=raw_date_format
    )

    # Check blanks and data types
    print("DataFrame info:")
    display(df.info())

    # Set datetime index
    df = df.set_index(date_column_name)
    df = df.sort_index()
    df.rename_axis("Date", inplace=True)

    # Check for missing timestamps before resampling using inferred source granularity.
    # This avoids brittle assumptions when a file contains mixed intervals over time.
    inferred_granularity = infer_dominant_granularity(df)

    if inferred_granularity is not None:
        expected = pd.date_range(
            start=expected_first_timestamp,
            end=expected_last_timestamp,
            freq=inferred_granularity
        )

        missing_dates = expected.difference(df.index)
        _summarize_datetime_index("Missing dates before resampling", missing_dates)

    else:
        print("Missing-date pre-check skipped: unable to infer source granularity.")

    # Handle mixed granularities
    df = handle_varying_granularities(df, output_granularity, keep_cols_info)

    # Check again after resampling
    expected_after = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=output_granularity
    )

    missing_after = expected_after.difference(df.index)
    _summarize_datetime_index("Missing dates after resampling", missing_after)

    is_strictly_sorted = df.index.is_unique and df.index.is_monotonic_increasing
    print(f"Chronological: {is_strictly_sorted}")

    # Filter to desired date range
    df = df[(df.index >= filter_first_timestamp) & (df.index <= filter_last_timestamp)]
    df = df.reset_index()


    # Final check
    expected_final= pd.date_range(
        start=filter_first_timestamp,
        end=(filter_last_timestamp),
        freq=output_granularity
    )
    print("Actual length:",len(df), "Expected length:",len(expected_final))

    # Rename keep cols
    rename_map = {}

    for col in df.columns:
        config = keep_cols_info.get(col)

        if config is None:
            continue

        rename_map[col] = config["rename"]

    df = df.rename(columns=rename_map)

    return df