import numpy as np
import pandas as pd

def parse_ppg_string(ppg_str):
    try:
        return np.array(
            [float(x) for x in ppg_str.strip().split()],
            dtype=np.float32
        )
    except:
        return None


def build_ppg_segments(df, segment_len=50, max_per_patient=30):
    df.columns = df.columns.str.strip()

    ppg_col = None
    for c in df.columns:
        if "ppg" in c.lower():
            ppg_col = c
            break
    if ppg_col is None:
        raise ValueError("No PPG column found")

    X_ppg, X_clin, y, pids = [], [], [], []

    for _, row in df.iterrows():
        signal = parse_ppg_string(row[ppg_col])
        if signal is None or len(signal) < segment_len:
            continue

        n = len(signal) // segment_len
        segments = signal[:n * segment_len].reshape(n, segment_len)
        segments = segments[np.std(segments, axis=1) > 1e-6]

        if len(segments) == 0:
            continue

        if len(segments) > max_per_patient:
            idx = np.random.choice(len(segments), max_per_patient, replace=False)
            segments = segments[idx]

        clin = np.array([
            row["AGE"],
            row["GENDER"],
            row["is_diabetic"],
            row["has_high_cholesterol"],
            row["is_obese"]
        ], dtype=np.float32)

        if pd.isna(row["CAD_LABEL"]):
            continue
        label = int(row["CAD_LABEL"])

        pid = row["SUBJECT_ID"]

        for seg in segments:
            X_ppg.append(seg[:, None])
            X_clin.append(clin)
            y.append(label)
            pids.append(pid)

    print("✅ Total segments extracted:", len(X_ppg))

    return (
        np.array(X_ppg, dtype=np.float32),
        np.array(X_clin, dtype=np.float32),
        np.array(y, dtype=np.int32),
        np.array(pids)
    )
