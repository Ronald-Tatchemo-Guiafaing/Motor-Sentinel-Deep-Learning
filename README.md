# Motor Sentinel — Deep Learning Fault Detection

**Author:** Ronald Tatchemo Guiafaing  
**Stack:** MATLAB · Simulink · Deep Learning Toolbox

Induction motor fault detection pipeline: Simulink modeling, dataset generation (3000 samples), spectrogram analysis, feature engineering, and comparison of MLP / CNN / LSTM classifiers with Simulink deployment.

## Structure

| Folder | Content |
|--------|---------|
| `01_Rapport/` | LaTeX report + figures |
| `02_TP1_Simulink/` | Async motor Simulink model (`.slx`) |
| `03_TP2_Dataset/` | Dataset generation scripts |
| `04_TP3_Spectrogrammes/` | Signal processing & spectrograms |
| `05_TP4_Features/` | Feature engineering (RMS, kurtosis, peaks) |
| `06_TP5_Modelisation/` | MLP, CNN, LSTM training |
| `07_TP6_Deploiement/` | CNN detector + Simulink integration |
| `08_Donnees/` | Generated `.mat` files (not in repo — see below) |

## Quick start

1. Open MATLAB R2019b or newer.
2. `cd` to this repository root.
3. Run `03_TP2_Dataset/generer_dataset.m` to regenerate datasets.
4. Follow TPs 3–6 in order.

## Note on datasets

Large `.mat` files (`Dataset_Moteur.mat` ~448 MB) are excluded from GitHub. Regenerate locally using the TP2 scripts.

## License

Academic project, 2025 to 2026.
