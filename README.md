# AI-Healthcare-Jetson

Tiny ECG classification experiments and Jetson benchmarking utilities for an
AI healthcare workflow. The current training code builds a compact PyTorch
`TinyResECG` model for binary ECG rhythm classification.

New models:
https://drive.google.com/drive/folders/1pRsft8d7OtFvDBFdMqmD6_H1VEFsl5Id?usp=sharing

## Environment Setup

The local development environment used for this project is the conda
environment named `dnn_env`. A focused list of project dependencies from that
environment is stored in `requirements.txt`.

Create and activate the environment:

```bash
conda create -n dnn_env python=3.10
conda activate dnn_env
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

On Jetson devices, PyTorch is often installed from NVIDIA JetPack-specific
wheels rather than standard PyPI. The captured environment uses:

```text
torch==2.5.0a0+872d972e41.nv24.8
```

If `pip install -r requirements.txt` cannot resolve that exact `torch` build,
install the PyTorch wheel that matches your JetPack/L4T version first, then
install the remaining requirements.

The preprocessing script `Latest/create_patient_windows.py` requires `wfdb`.
The training script can optionally use `thop` for FLOP profiling if it is
installed.

## Project Layout

```text
.
|-- README.md
|-- requirements.txt
|-- Latest/
|   |-- create_patient_windows.py
|   |-- ecg_dataset.py
|   |-- model.py
|   |-- train.py
|   |-- tinyresecg_v3.pth
|   |-- *_metrics.csv
|   |-- *confusion_matrix.csv
|   `-- training_history.csv
|-- Benchmark/
|   |-- benchmark_tinyresecg.py
|   |-- BENCHMARK_README.md
|   `-- benchmark_results/
|       |-- benchmark_report.md
|       |-- benchmark_report.json
|       |-- layer_breakdown.csv
|       `-- tegrastats.log
|-- Archive/
|   |-- ECG_pipeline.ipynb
|   |-- test.ipynb
|   |-- *.h5
|   `-- *.tflite
|-- mit-bih-arrhythmia-database-1.0.0/
`-- mit-bih-noise-stress-test-database-1.0.0/
```

## Directory Details

- `Latest/`: active PyTorch dataset, model, preprocessing, training scripts,
  trained checkpoint, and generated metrics.
- `Benchmark/`: Jetson inference benchmark script and benchmark result reports.
- `Archive/`: older notebooks and exported Keras/TFLite model artifacts kept
  for reference.
- `mit-bih-arrhythmia-database-1.0.0/`: local MIT-BIH Arrhythmia Database
  files.
- `mit-bih-noise-stress-test-database-1.0.0/`: local MIT-BIH Noise Stress Test
  Database files.

## Common Commands

Run preprocessing after setting the raw and output paths in
`Latest/create_patient_windows.py`:

```bash
conda activate dnn_env
python Latest/create_patient_windows.py
```

Train the model after setting the data and output paths in `Latest/train.py`:

```bash
conda activate dnn_env
python Latest/train.py
```

Run the Jetson benchmark from the `Benchmark/` directory:

```bash
conda activate dnn_env
cd Benchmark
python benchmark_tinyresecg.py --checkpoint ../Latest/tinyresecg_v3.pth
```
