Task 3: Data preprocessing and Transformer training

Contains:

- `data_processing.py` — preprocess cleaned NASA battery dataset into binned main CSV and physics CSV.
- `transformer_model.py` — PyTorch Transformer model.
- `train_transformer.py` — training script with physics-constrained losses.

## Data Processing

Automatically downloads the NASA battery dataset from Kaggle (228 MB, ~2 min to download + extract):

```powershell
python data_processing.py --download --out_dir data_out
```

Or use a local path:

```powershell
python data_processing.py --dataset_path "C:/path/to/cleaned_dataset" --out_dir data_out
```

Output:
- `data_out/battery_main_binned20.csv` — 20 bins per discharge cycle (fast training)
- `data_out/battery_phys_resampled200.csv` — 200 time points per cycle (physics losses)

## Training

```powershell
python -m pip install -r requirements.txt
python train_transformer.py --main_csv data_out/battery_main_binned20.csv --phys_csv data_out/battery_phys_resampled200.csv --epochs 100 --batch_size 32 --device cpu
```

The transformer predicts: **[SoH, Q_Ah, E_Wh]** with physics-constrained losses (λ_q=0.3, λ_e=0.3).

Model saves to `models/transformer_soh.pt`.

