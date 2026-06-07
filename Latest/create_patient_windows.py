import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

# =============================================================================
# CONFIG — edit paths and thresholds here
# =============================================================================

# Root directory that contains the raw LTAFDB record files
# (each record has  <id>.dat  <id>.hea  <id>.atr)
LTAFDB_RAW_DIR = "/ssd_scratch/abnp/ecg_on_edge/data/ltafdb"

# Output directory for windowed .npy files
OUTPUT_DIR = "/ssd_scratch/abnp/ecg_on_edge/data/binary_class/windowed"

# Sampling rate of LTAFDB
FS = 128  # Hz

# Window and step sizes (samples)
WINDOW_SIZE = 10 * FS   # 1280 samples = 10 s
STEP_SIZE   = 5  * FS   # 640 samples  = 5 s  (50% overlap)

# Bandpass filter cutoffs (Hz)
BP_LOW  = 0.5
BP_HIGH = 40.0
BP_ORDER = 4

# Label assignment thresholds
# A window must have ≥ this fraction of a single class to be labelled
AFIB_THRESHOLD   = 0.50   # ≥ 50% AFIB samples → AFIB
NORMAL_THRESHOLD = 0.50   # ≥ 50% Normal samples → Normal

# Set True to skip windows that contain any non-Normal/non-AFIB rhythm
# Set False to assign label by majority vote even for mixed windows
SKIP_MIXED = True

# LTAFDB rhythm annotation strings
# Rhythm annotations use the 'aux_note' field in the .atr file
AFIB_RHYTHMS   = {"(AFIB"}      # AFIB annotation token
NORMAL_RHYTHMS = {"(N"}         # Normal sinus rhythm token
# Other rhythms present in LTAFDB: (AFL (atrial flutter), (J (junctional) etc.
# These are skipped when SKIP_MIXED=True

# =============================================================================
# All patient record IDs in LTAFDB
# =============================================================================

ALL_PATIENTS = [
    "00", "01", "03", "05", "06", "07", "08", "10", "11", "13",
    "15", "16", "17", "18", "19", "20", "21", "22", "23", "24",
    "25", "26", "28", "30", "32", "33", "34", "35", "37", "38",
    "39", "42", "43", "44", "45", "47", "48", "49", "51", "53",
    "54", "55", "56", "58", "60", "62", "64", "65", "68", "69",
    "70", "71", "72", "74", "75", "100", "101", "102", "103",
    "104", "105", "110", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "120", "121", "122"
]

# =============================================================================
# Patient-wise split (same as training script — defined here for reference)
# =============================================================================

TRAIN_PATIENTS = [
    "55","39","117","33","35","111","07","56","115","16","120",
    "00","01","03","05","06","08","10","100","101","102","103",
    "104","105","110","112","113","114","116","119","121","122",
    "13","15","17","18","19","20","21","22","23","24","25","26",
    "28","30","32","34","37","38","60","64","65","68","69","70","71","75"
]

VALID_PATIENTS = ["45","51","44","49","42","43","47","48"]

TEST_PATIENTS  = ["118","11","72","74","62","53","54","58"]


# =============================================================================
# Bandpass filter
# =============================================================================

def bandpass_filter(signal: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
   
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    filtered = np.empty_like(signal)
    for ch in range(signal.shape[1]):
        filtered[:, ch] = filtfilt(b, a, signal[:, ch])
    return filtered


# =============================================================================
# Per-window z-score normalisation
# =============================================================================

def zscore_normalise(window: np.ndarray, eps: float = 1e-8) -> np.ndarray:

    mean = window.mean(axis=0, keepdims=True)
    std  = window.std(axis=0, keepdims=True)
    return (window - mean) / (std + eps)


# =============================================================================
# Build a sample-level label array from WFDB rhythm annotations
# =============================================================================

def build_sample_labels(record_len: int, annotation) -> np.ndarray:
   
    labels = np.full(record_len, -1, dtype=np.int8)

    # Filter to rhythm-change rows only (symbol='+', non-empty aux_note).
    # Beat-level rows (symbol='N','V', etc.) have empty aux_note and must
    # be skipped — iterating them would reset every label back to -1.
    rhythm_samples = []
    rhythm_notes   = []

    for samp, sym, note in zip(annotation.sample, annotation.symbol, annotation.aux_note):
        note_clean = note.strip("\x00").strip()
        if sym == "+" and note_clean:
            rhythm_samples.append(samp)
            rhythm_notes.append(note_clean)

    # Fallback: if no '+' markers found, try any non-empty aux_note row
    if not rhythm_samples:
        for samp, note in zip(annotation.sample, annotation.aux_note):
            note_clean = note.strip("\x00").strip()
            if note_clean and not note_clean.startswith("\x01"):
                rhythm_samples.append(samp)
                rhythm_notes.append(note_clean)

    for i, (samp, note) in enumerate(zip(rhythm_samples, rhythm_notes)):
        if note in AFIB_RHYTHMS:
            lbl = 1
        elif note in NORMAL_RHYTHMS:
            lbl = 0
        else:
            lbl = -1   # VT, AFL, or any other rhythm — skipped downstream

        # Propagate this label forward until the next rhythm change
        end = rhythm_samples[i + 1] if i + 1 < len(rhythm_samples) else record_len
        labels[samp:end] = lbl

    return labels



# =============================================================================
# Process a single patient record
# =============================================================================

def process_patient(patient_id: str) -> tuple[np.ndarray, np.ndarray]:
  
    record_path = os.path.join(LTAFDB_RAW_DIR, patient_id)

    # ------------------------------------------------------------------
    # 1. Load raw signal  (N_samples, 2)
    # ------------------------------------------------------------------
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal.astype(np.float32)  # (N, 2)

    if signal.ndim == 1:
        signal = signal[:, np.newaxis]            # guard for single-lead edge case

    n_samples, n_channels = signal.shape

    # ------------------------------------------------------------------
    # 2. Replace NaN/Inf that occasionally appear in LTAFDB
    # ------------------------------------------------------------------
    nan_mask = ~np.isfinite(signal)
    if nan_mask.any():
        n_bad = nan_mask.sum()
        print(f"    [{patient_id}] Replacing {n_bad} non-finite sample(s) with 0")
        signal[nan_mask] = 0.0

    # ------------------------------------------------------------------
    # 3. Bandpass filter — whole record at once for continuity
    # ------------------------------------------------------------------
    signal = bandpass_filter(signal, FS, BP_LOW, BP_HIGH, BP_ORDER)

    # ------------------------------------------------------------------
    # 4. Load rhythm annotations
    # ------------------------------------------------------------------
    try:
        annotation = wfdb.rdann(record_path, "atr")
    except Exception as exc:
        raise RuntimeError(
            f"Could not read annotations for patient {patient_id}: {exc}"
        )

    sample_labels = build_sample_labels(n_samples, annotation)

    # ------------------------------------------------------------------
    # 5. Slide window and assign labels
    # ------------------------------------------------------------------
    X_list = []
    y_list = []

    n_windows = 0
    n_skipped_mixed = 0
    n_skipped_short  = 0

    start = 0
    while start + WINDOW_SIZE <= n_samples:
        end = start + WINDOW_SIZE

        window_labels = sample_labels[start:end]   # (1280,)

        n_afib   = int((window_labels == 1).sum())
        n_normal = int((window_labels == 0).sum())
        n_other  = int((window_labels == -1).sum())
        total    = WINDOW_SIZE

        frac_afib   = n_afib   / total
        frac_normal = n_normal / total
        frac_other  = n_other  / total

        # Determine window label
        if frac_other > 0 and SKIP_MIXED:
            # Any non-Normal/AFIB samples → skip
            n_skipped_mixed += 1
            start += STEP_SIZE
            continue

        if frac_afib >= AFIB_THRESHOLD:
            label = 1
        elif frac_normal >= NORMAL_THRESHOLD:
            label = 0
        else:
            # Neither class dominates — skip ambiguous window
            n_skipped_mixed += 1
            start += STEP_SIZE
            continue

        # Extract and normalise the window
        window = signal[start:end, :]               # (1280, 2)
        window = zscore_normalise(window)

        X_list.append(window)
        y_list.append(label)
        n_windows += 1

        start += STEP_SIZE

    if len(X_list) == 0:
        print(f"    [{patient_id}] WARNING: no valid windows produced!")
        return np.empty((0, WINDOW_SIZE, n_channels), dtype=np.float32), np.empty(0, dtype=np.int64)

    X = np.stack(X_list, axis=0).astype(np.float32)   # (N, 1280, 2)
    y = np.array(y_list, dtype=np.int64)               # (N,)

    n_afib_wins   = int((y == 1).sum())
    n_normal_wins = int((y == 0).sum())

    print(
        f"    [{patient_id}] "
        f"total={len(X):5d}  Normal={n_normal_wins:5d}  AFIB={n_afib_wins:5d}  "
        f"skipped_mixed={n_skipped_mixed}  duration={n_samples/FS/60:.1f} min"
    )

    return X, y


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("LTAFDB windowing — binary classification (Normal vs AFIB)")
    print(f"  Window : {WINDOW_SIZE} samples ({WINDOW_SIZE/FS:.0f} s)")
    print(f"  Step   : {STEP_SIZE}   samples ({STEP_SIZE/FS:.0f} s overlap)")
    print(f"  Filter : {BP_LOW}–{BP_HIGH} Hz Butterworth order {BP_ORDER}")
    print(f"  Output : {OUTPUT_DIR}")
    print("=" * 70)

    # Track split-level stats for a quick sanity check at the end
    split_stats = {
        "TRAIN": {"Normal": 0, "AFIB": 0},
        "VALID": {"Normal": 0, "AFIB": 0},
        "TEST":  {"Normal": 0, "AFIB": 0},
    }

    def get_split(pid):
        if pid in TRAIN_PATIENTS: return "TRAIN"
        if pid in VALID_PATIENTS: return "VALID"
        if pid in TEST_PATIENTS:  return "TEST"
        return "UNASSIGNED"

    failed = []

    for patient_id in ALL_PATIENTS:
        x_path = os.path.join(OUTPUT_DIR, f"{patient_id}_X.npy")
        y_path = os.path.join(OUTPUT_DIR, f"{patient_id}_y.npy")

        # Skip if already processed (remove this block to always reprocess)
        if os.path.exists(x_path) and os.path.exists(y_path):
            print(f"  [{patient_id}] already exists — skipping")
            X = np.load(x_path, mmap_mode="r")
            y = np.load(y_path, mmap_mode="r")
        else:
            try:
                X, y = process_patient(patient_id)
            except FileNotFoundError:
                print(f"  [{patient_id}] record not found — skipping")
                failed.append(patient_id)
                continue
            except Exception as exc:
                print(f"  [{patient_id}] ERROR: {exc}")
                failed.append(patient_id)
                continue

            if len(X) == 0:
                failed.append(patient_id)
                continue

            np.save(x_path, X)
            np.save(y_path, y)

        # Accumulate split stats
        split = get_split(patient_id)
        if split in split_stats and len(y) > 0:
            split_stats[split]["Normal"] += int((y == 0).sum())
            split_stats[split]["AFIB"]   += int((y == 1).sum())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)
    for split, counts in split_stats.items():
        normal = counts["Normal"]
        afib   = counts["AFIB"]
        total  = normal + afib
        if total == 0:
            print(f"  {split:7s}: (no data)")
            continue
        print(
            f"  {split:7s}: total={total:6d}  "
            f"Normal={normal:6d} ({100*normal/total:.1f}%)  "
            f"AFIB={afib:6d} ({100*afib/total:.1f}%)"
        )

    if failed:
        print(f"\nFailed / missing patients: {failed}")

    print("\nDone.")


if __name__ == "__main__":
    main()
