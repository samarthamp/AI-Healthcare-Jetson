import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
   

    def __init__(
        self,
        patient_list: list,
        data_dir: str,
        is_train: bool = False,
    ):
        self.data_dir = data_dir
        self.is_train = is_train

        self.index = []
        self.data_cache = {}

        # ------------------------------------------------------------------
        # Load all patient files into memory-mapped cache
        # ------------------------------------------------------------------
        for patient in patient_list:
            X_path = os.path.join(data_dir, f"{patient}_X.npy")
            y_path = os.path.join(data_dir, f"{patient}_y.npy")

            X = np.load(X_path, mmap_mode="r")
            y = np.load(y_path, mmap_mode="r")

            assert len(X) == len(y), (
                f"Sample/label count mismatch for patient {patient}: "
                f"{len(X)} vs {len(y)}"
            )

            print(f"  Patient {patient}: {len(X)} windows")

            self.data_cache[patient] = (X, y)

            for idx in range(len(X)):
                self.index.append((patient, idx))

        print(f"  Total indexed samples: {len(self.index)}\n")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.index)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        patient, sample_idx = self.index[idx]

        X, y = self.data_cache[patient]

        # (window_len, 2)  →  copy so augmentation writes are safe
        signal = torch.from_numpy(np.array(X[sample_idx])).float()

        # (2, window_len)  — channel-first for Conv1d
        signal = signal.permute(1, 0)

        label = torch.tensor(int(y[sample_idx]), dtype=torch.long)

        if self.is_train:
            signal = self._augment(signal)

        return signal, label

    # ------------------------------------------------------------------
    # Augmentation pipeline
    # ------------------------------------------------------------------
    def _augment(self, signal: torch.Tensor) -> torch.Tensor:
       
        C, T = signal.shape

        # 1. Random amplitude scaling  [prob=0.7]
        #    Simulates variable electrode contact / patient impedance
        if torch.rand(1).item() < 0.70:
            scale = torch.empty(C, 1).uniform_(0.75, 1.25)
            signal = signal * scale

        # 2. Additive Gaussian noise  [prob=0.6]
        #    Simulates muscle artefact / baseline noise
        if torch.rand(1).item() < 0.60:
            snr_db = torch.empty(1).uniform_(20, 35).item()
            signal_power = signal.pow(2).mean()
            noise_power = signal_power / (10 ** (snr_db / 10) + 1e-8)
            noise = torch.randn_like(signal) * noise_power.sqrt()
            signal = signal + noise

        # 3. Baseline wander (low-freq sinusoidal drift)  [prob=0.4]
        #    Simulates respiration artefact
        if torch.rand(1).item() < 0.40:
            freq = torch.empty(1).uniform_(0.05, 0.5).item()   # Hz
            phase = torch.empty(1).uniform_(0, 2 * 3.14159).item()
            amplitude = torch.empty(1).uniform_(0.02, 0.10).item()
            t = torch.linspace(0, T / 128, T)                  # 128 Hz
            wander = (amplitude * torch.sin(2 * 3.14159 * freq * t + phase))
            signal = signal + wander.unsqueeze(0)

        # 4. Random time shift  [prob=0.5]
        #    Roll the signal along time axis (circular), simulates onset offset
        if torch.rand(1).item() < 0.50:
            shift = int(torch.randint(-64, 64, (1,)).item())   # ±0.5 s at 128 Hz
            signal = torch.roll(signal, shifts=shift, dims=1)

        # 5. Channel dropout  [prob=0.2]
        #    Randomly zero one lead to teach robustness to single-lead failures
        if torch.rand(1).item() < 0.20:
            ch = torch.randint(0, C, (1,)).item()
            signal[ch] = 0.0

        # 6. Random polarity flip per channel  [prob=0.3]
        #    ECG lead polarity can vary with electrode placement
        if torch.rand(1).item() < 0.30:
            flip_mask = (torch.rand(C) > 0.5).float() * 2 - 1   # ±1
            signal = signal * flip_mask.unsqueeze(1)

        return signal
