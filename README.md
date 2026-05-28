# TC-SGDN: Digital Twin-Assisted UAV Fault Diagnosis

> **T**win-**C**onsistent **S**patio-**T**emporal **G**raph **D**iagnosis **N**etwork — A graph attention network for online fault diagnosis of multirotor UAVs, leveraging digital twin residuals to enhance weak fault observability.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Dataset Description](#dataset-description)
- [Fault-Injection Protocol](#fault-injection-protocol)
- [Physics-Informed Baseline Comparison](#physics-informed-baseline-comparison)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Hardware & Software Platform](#hardware--software-platform)

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────── ┐
│                   Real-Time Fault Diagnosis Pipeline                  │
│                                                                       │
│  ┌──────────┐     ┌───────────┐    ┌──────────────┐     ┌──────────┐  │
│  │ Physical │───▶│  Digital   │───▶│   Feature   │───▶│  TS-SGDN │  │
│  │   UAV    │     │   Twin    │    │ Construction │     │ Diagnoser│  │
│  │  (IMU)   │     │(CopterSim)│    │  (18-ch)     │     │          │  │
│  └──────────┘     └───────────┘    └──────────────┘     └──────────┘  │
│       │               │                │                    │         │
│   raw_data(9ch)   sim_data(9ch)   err = raw - sim       4-class       │
│   [Acc,Gyro,Mag]  [Acc,Gyro,Mag]  → 18ch: raw‖err      diagnosis      │
└──────────────────────────────────────────────────────────────────────┘
```

The pipeline consists of four stages:

1. **Physical UAV** collects 9-channel IMU data (Accelerometer × 3, Gyroscope × 3, Magnetometer × 3) at 50 Hz.
2. **Digital Twin** (CopterSim) runs a synchronized high-fidelity simulation, producing an identical 9-channel output.
3. **Feature Construction** computes the DT residual (`err = raw − sim`) and interleaves it with the raw signal to form an 18-channel representation.
4. **TC-SGDN** constructs a spatio-temporal sensor graph and performs fault classification via graph attention + bidirectional GRU.

### Communication Modes

| Mode | Abbreviation | Description |
|------|-------------|-------------|
| V2V | Vehicle-to-Vehicle | Direct data link between physical UAV and DT |
| R2V | Remote-to-Vehicle | Remote ground station relaying data to DT |

---

## Repository Structure

```
DT-Diagnosis/
├── data/                         # Base flight data (Draw → Dsim → Dpro)
│   ├── Draw/                     #   Raw sensor recordings
│   ├── Dsim/                     #   DT simulation outputs
│   ├── Dpro/                     #   Preprocessed & synchronized data
│   └── Dpro.py                   #   Preprocessing script
│
├── data_for_v2v_c4/              # V2V c4 dataset (training & in-domain test)
│   ├── Draw/ → Dsim/ → Dpro/
│   └── Dpro.py
├── data_for_r2v_c4/              # R2V c4 dataset
│   ├── Draw/ → Dsim/ → Dpro/
│   └── Dpro.py
├── data_for_v2v_m6/              # V2V m6 dataset (cross-domain generalization)
│   ├── Draw/ → Dsim/ → Dpro/
│   └── Dpro.py
├── data_ver/                     # R2V m6 dataset (cross-domain generalization)
│   ├── Draw/ → Dsim/ → Dpro/
│   └── Dpro.py
│
├── model/                        # Training notebooks for all classifiers
│   ├── Classifier_MGAT_GRU.ipynb #   TS-SGDN (proposed method)
│   ├── Classifier_OGAT_GRU.ipynb #   Ablation: standard GAT + GRU
│   ├── Classifier_OGCN_GRU.ipynb #   Ablation: GCN + GRU
│   ├── Classifier_CNN_LSTM.ipynb  #   Baseline: CNN-LSTM
│   ├── Classifier_OGAT_LSTM.ipynb #   Ablation: GAT + LSTM
│   ├── Classifier_OGCN_GRU_S3.ipynb # Ablation: 3-sensor GCN + GRU
│   ├── Classifier_FFSAN.ipynb     #   Baseline [1]: FFSAN
│   ├── Classifier_SAE_CNN.ipynb   #   Baseline [2]: SAE-CNN
│   ├── Classifier_CNN_Transformer.ipynb # Baseline [3]: CNN-Transformer
│   └── Classifier_WaveletCNN.ipynb#   Baseline [4]: WaveletCNN
│
├── ver/                          # Online verification & deployment
│   ├── src/r2v/
│   │   ├── ver_v2v_online.py     #   Online V2V/R2V diagnosis script
│   │   ├── db_FD.json            #   Fault diagnosis config
│   │   ├── utilits/              #   Model definitions for deployment
│   │   ├── QuadModelSITL.bat     #   SITL simulation launcher
│   │   └── SITLRun.bat           #   SITL execution script
│   └── include/                  #   Shared utilities
│
└── README.md                     # This file
```

### Data Pipeline (`Draw` → `Dsim` → `Dpro`)

Each dataset directory follows the same 3-stage pipeline:

| Stage | Directory | Description |
|-------|-----------|-------------|
| **Draw** | `Draw/` | Raw flight recordings from physical UAV (or HIL simulation) |
| **Dsim** | `Dsim/` | Corresponding digital twin simulation outputs |
| **Dpro** | `Dpro/` | Synchronized, time-aligned, preprocessed CSV files |
| **Script** | `Dpro.py` | Automated preprocessing: synchronization + error computation |

---

## Dataset Description

> 📂 Detailed dataset documentation including channel layout, preprocessing pipeline, and graph construction is available in [`data_for_v2v_c4/README.md`](data_for_v2v_c4/README.md).

### Sensor Configuration

9-channel synchronized IMU data at **50 Hz** sampling rate:

| Channel Group | Channels | Unit |
|:---:|:---:|:---:|
| Accelerometer | Acc-x, Acc-y, Acc-z | m/s² |
| Gyroscope | Gyro-x, Gyro-y, Gyro-z | rad/s |
| Magnetometer | Mag-x, Mag-y, Mag-z | Gauss |

### 4-Class Fault Taxonomy

| Class | Label | Fault Type | Physical Mechanism | Injection |
|:-----:|:-----:|-----------|-------------------|-----------|
| C1 | 1 | Normal | No fault | — |
| C2 | 2 | Accelerometer noise interference (ANI) | Acc-x channel interference | PX4 fault ID 12, flag 3, firmware trigger parameter `A_acc=3000` |
| C3 | 3 | One motor signal jump (OMSJ) | Motor #3 actuator/control signal jump | PX4 fault ID 11, flag 3, `Delta u=200` |
| C4 | 4 | One motor power drop (OMPD) | Motor #3 multiplicative efficiency degradation | PX4 fault ID 11, flag 4, `eta_eff=0.85` |

### Fault-Injection Protocol

The benchmark uses a bounded PX4-triggered fault-emulation protocol. Each fault run follows the same safety-oriented operating sequence:

1. Arm and take off.
2. Record the takeoff segment (`mode2_3`).
3. Enter hover stabilization (`mode2_6`).
4. Trigger the target fault (`mode2_9`).
5. Stop the fault-recording phase with a stop tag.
6. Enter landing/recovery (`mode2_7`).
7. Restore the fault parameter to its nominal value before the next run.

The flight-log segmentation is controlled by PX4 command markers written into `vehicle_command_0.csv`. The preprocessing utility `RflyDtrain.py` identifies start/stop markers (`param7=666/777`) and extracts the corresponding segments. The downstream `Dpro.py` pipeline aligns the real UAV data from `Draw/` with the synchronized DT data from `Dsim/`, interpolates unequal-length sequences, and writes synchronized raw, simulated, and residual files to `Dpro/`.

| Class | Fault mechanism | Severity setting | Primary affected channel | Trigger duration | Operating window | Reset / recovery procedure |
|:---:|---|---|---|---:|---|---|
| C2 / ANI | Accelerometer noise interference, PX4 fault ID 12, flag 3 | Firmware-level Acc-x interference parameter `A_acc=3000`. This value is retained as the PX4 trigger parameter unless a firmware-to-physical unit conversion is specified. | Directly applied to `accelerometer_m_s2[0]` (Acc-x); all IMU/magnetometer channels are retained as observed responses after dynamic coupling. | ~10.00 s over 10 runs | Triggered after takeoff and a ~3 s hover-stabilization segment. | The fault window is terminated by `2,10,2,9`; the vehicle then enters landing/recovery. Before the next run, the fault parameter is restored to `A_acc=0`. |
| C3 / OMSJ | One-motor signal jump, PX4 fault ID 11, flag 3 | Motor #3 signal-jump/control perturbation parameter `Delta u=200`. | Directly applied to the Motor #3 actuator/control channel; the response propagates to gyro/acceleration residuals through actuator-body coupling. | ~10.00 s over 10 runs | Triggered after takeoff and a ~3 s hover-stabilization segment. | The fault window is terminated by `2,10,2,9`; the vehicle then enters landing/recovery. Before the next run, the motor perturbation is disabled by setting `Delta u=0`. |
| C4 / OMPD | One-motor power drop, PX4 fault ID 11, flag 4 | Motor #3 efficiency scaling `eta_eff=0.85`, corresponding to a 15% efficiency/power reduction. | Directly applied to the Motor #3 efficiency/actuation output; the induced response is observed through synchronized IMU residuals. | ~5.00 s over 10 runs | Triggered after takeoff and a ~3 s hover-stabilization segment. | The fault window is terminated by `2,10,2,9`; the vehicle then enters landing/recovery. Before the next run, actuator efficiency is restored to `eta_eff=1`. |

Additional reproducibility notes:

- Fault-mode directories encode the target channel and trigger parameter, e.g., `3-acc-x-axis-flag3-3000`, `6-motor03-flag3-200`, and `5-motor03-flag4-085`.
- The model input keeps nine synchronized sensor channels: `gyro_rad[0,1,2]`, `accelerometer_m_s2[0,1,2]`, and `magnetometer_ga[0,1,2]`.
- Residual files are computed from synchronized physical and virtual data. In the paper, this residual is used as the physical-virtual consistency cue for TC-SGDN.
- C4 is intentionally retained as a challenging case. Unlike C2/C3, its multiplicative efficiency degradation can look like a scaled nominal response, which explains its larger overlap with normal conditions and the weaker separability of additive residuals.

### Dataset Variants

| Dataset | Path | Comm. Mode | Purpose |
|---------|------|:----------:|---------|
| **c4** | `data_for_v2v_c4/` | V2V | Training & in-domain test |
| **c4** | `data_for_r2v_c4/` | R2V | Training & in-domain test |
| **m6** | `data_for_v2v_m6/` | V2V | Cross-domain generalization (aggressive maneuvers) |
| **m6** | `data_ver/` | R2V | Cross-domain generalization (aggressive maneuvers) |

---

## Physics-Informed Baseline Comparison

To clarify how TC-SGDN differs from conventional physics-informed machine learning baselines, we provide a controlled PI-ResGNN comparison. PI-ResGNN uses the same 18-channel physical/DT residual input, the same GAT backbone, and the same BiGRU temporal encoder as TC-SGDN. The only controlled difference is that PI-ResGNN replaces the DT-consistency graph with generic graph topologies.

### Controlled Baseline Design

| Method | Physics-informed residual input | GNN + temporal encoder | Graph topology / semantics |
|---|:---:|:---:|---|
| TC-SGDN | Yes | GAT + BiGRU | DT-consistency graph with temporal-continuity, sensor-coupling, and physical-virtual correction edges |
| PI-ResGNN (Correlation) | Yes | GAT + BiGRU | Global Top-K cosine adjacency, K=5 |
| PI-ResGNN (Fully connected) | Yes | GAT + BiGRU | Intra-timestep fully connected sensor graph |
| PI-ResGNN (KNN) | Yes | GAT + BiGRU | Euclidean KNN adjacency, K=5 |

This comparison separates feature-level physics-informed learning from graph-native DT-consistency learning. PI-ResGNN receives the same residual features as TC-SGDN, but it does not encode the synchronized DT as an online healthy-reference relation inside graph message passing.

### Main Results

Accuracy is reported below.

| Dataset | TC-SGDN | PI-ResGNN (Correlation) | PI-ResGNN (Fully connected) | PI-ResGNN (KNN) |
|---|---:|---:|---:|---:|
| DS1 v2v-c4 | 0.9819 | 0.9801 | 0.9543 | 0.9796 |
| DS2 r2v-c4 | 0.9965 | 0.8715 | 0.8332 | 0.9936 |
| DS4 r2v-m6 | 0.9822 | 0.8865 | 0.7589 | 0.7839 |

In the matched DS1 setting, physics-informed residual features already provide strong observability, so PI-ResGNN variants can approach TC-SGDN. Under DS2 distribution shift and DS4 unseen high-maneuver conditions, generic correlation, fully connected, and KNN graph structures degrade more clearly. This indicates that residual features alone are not sufficient for robust UAV fault diagnosis; the key contribution of TC-SGDN is to embed physical-virtual consistency into graph topology and message propagation.

### Reproduction Configuration

| Item | Setting |
|---|---|
| Input channels | 18 channels: Acc, ErrAcc, Gyro, ErrGyro, Mag, ErrMag |
| Sequence length | 80 |
| Sliding step | 4 |
| Graph nodes | 6 sensor/residual nodes per timestep, 480 nodes per sample |
| Node feature dimension | 3 axes per node |
| Backbone | GAT, hidden dimension 64, 4 attention heads |
| Temporal encoder | Bidirectional GRU, hidden dimension 128 |
| Classifier | Mean pooling + fully connected layer |
| Optimizer | Adam, learning rate 0.001, weight decay 1e-4 |
| Batch size | 32 |
| Training epochs | 15 |
| K for Correlation/KNN graphs | 5 |
---

## Model Architecture

### TS-SGDN (Proposed Method)
```
Input: (T=80, 18) time-series
    │
    ▼
┌─────────────────────────────────────────┐
│      Spatio-Temporal Graph Construction │
│  • 6 sensor nodes per timestep          │
│  • 3 features per node (x, y, z)        │
│  • 480 nodes total (6 × 80)             │
│  • 6 edge types (temporal/spatial/R↔E)  │
└──────────────┬──────────────────────────┘
               ▼
┌───────────────────────────────────────────┐
│        Edge-Type-Aware GAT (MyGATConv)    │
│  • in_channels = 3                        │
│  • out_channels = 32                      │
│  • heads = 4 (multi-head, concat=True)    │
│  • Edge-type attention weighting:         │
│    w_temporal(0) = 0.1                    │
│    w_spatial(1-4) = 0.2                   │
│    w_raw↔err(5)  = 0.7                    │
│  • Additive bias: α = α·w + 1.0           │
│  • LeakyReLU(0.2), Dropout(0.3)           │
│  → Output: (480, 128)                     │
└──────────────┬────────────────────────────┘
               ▼
┌───────────────────────────────────────────┐
│           Swish Activation                │
│  x = σ(x) × x                             │
│  → Gate-modulated feature refinement      │
└──────────────┬────────────────────────────┘
               ▼
┌───────────────────────────────────────────┐
│     Reshape to Temporal Sequence          │
│  (480, 128) → (80, 6×128) = (80, 768)     │
└──────────────┬────────────────────────────┘
               ▼
┌───────────────────────────────────────────┐
│        Bidirectional GRU                  │
│  • input_dim = 768                        │
│  • hidden_dim = 64                        │
│  • bidirectional = True                   │
│  • layers = 1                             │
│  → Output: (80, 128)                      │
└──────────────┬────────────────────────────┘
               ▼
┌───────────────────────────────────────────┐
│       Temporal Mean Pooling               │
│  (80, 128) → (128,)                       │
└──────────────┬────────────────────────────┘
               ▼
┌───────────────────────────────────────────┐
│        Fully Connected Classifier         │
│  Linear(128 → 4) → logits                 │
└───────────────────────────────────────────┘
```

### Sensor Graph Topology

```
Timestep t                    Timestep t+1
┌─────────────────┐          ┌─────────────────┐
│ Acc   ←──5──→ EAcc  │──0──▶│ Acc   ←──5──→ EAcc  │
│  ↑↕1,3          │          │  ↑↕              │
│ Gyro  ←──5──→ EGyro │──0──▶│ Gyro  ←──5──→ EGyro │
│  ↑↕2,4          │          │  ↑↕              │
│ Mag   ←──5──→ EMag  │──0──▶│ Mag   ←──5──→ EMag  │
└─────────────────┘          └─────────────────┘
  Numbers = edge type IDs
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 15 |
| Batch size | 64 |
| Loss function | CrossEntropyLoss |
| Train / Test split | 70% / 30% |
| Shuffle | True |
| Random seed | 42 (`torch.manual_seed`, `np.random.seed`, `random_state`) |
| Device | CUDA (GPU) |

---

## Hardware & Software Platform

### Hardware

| Component | Specification |
|-----------|--------------|
| UAV Platform | FEISI X150 ([feisilab.com](http://www.feisilab.com/?product_31/62.html)) |
| Onboard Compute | Rockchip RK3588 (8-core ARM, 6 TOPS NPU) |
| Ground Station GPU | NVIDIA GeForce RTX 3060 Laptop (6 GB GDDR6) |
| Indoor Positioning | Motion Capture System |
| State Estimation | 6-axis IMU (Accelerometer + Gyroscope) |

### Software

| Component | Link |
|-----------|------|
| DT Simulator | CopterSim ([GitHub](https://github.com/RflySim/CopterSim)) |
| 3D Renderer | RflySim3D ([GitHub](https://github.com/RflySim/Docs)) |
| Ground Control | QGroundControl ([qgroundcontrol.com](https://qgroundcontrol.com/)) |
| DL Framework | PyTorch + torch\_geometric |
| Python | 3.8+ |

### Dependencies

```
torch >= 1.10
torch_geometric
numpy
pandas
scikit-learn
scipy
matplotlib
openpyxl
```

---

---

## Robustness Analysis to DT Uncertainty

To evaluate the operational boundaries of the TC-SGDN framework, we systematically investigated its robustness against two primary sources of digital twin uncertainties: **parameter perturbations** and **synchronization errors**.

### Robustness to DT Parameter Perturbations

![fig3](figures/fig3.png)

**Figure 3. Robustness of the diagnosis model under DT feature perturbation.** Tested on a 4-class UAV fault diagnosis task (normal, X-axis Acc fault, motor efficiency fault, motor PWM limiting fault); perturbations are applied only to DT error feature channels, with reference noise standard deviation $\sigma=0.558$. (a)(b) Line plots of model performance under Gaussian noise ($0\sim5\sigma$) and systematic bias ($0\sim3\sigma$), respectively. (c) Bar chart of model accuracy under selective $3\sigma$ noise perturbation on Acc, Gyro, and Mag channels. (d) Scatter with quadratic fitting curve showing the correlation between noise level and accuracy. (e) Horizontal bar chart of model perturbation sensitivity. (f) Line plot of accuracy with increasing noise level.

#### Key Findings

*   **Random Noise Tolerance (Unmodeled Non-linearities):** The model exhibits strong robustness against random modeling uncertainties. Accuracy remains highly stable (>89%) even when random noise reaches $1\sigma$. Random errors fail to break the spatiotemporal invariants structure learned by the model, demonstrating that the graph network effectively absorbs non-systematic residuals.
*   **Systematic Bias Sensitivity:** The system is highly sensitive to systematic offsets. A mere $0.5\sigma$ bias causes accuracy to plummet to ~30%. This reveals that **"structural offsets"** (not complex unmodeled non-linearities) are the dominant risk factor. These offsets shift the healthy baseline and invalidate the physical meaning of the residual direction.
*   **Channel Dependencies:** Perturbations in accelerometer channels cause significantly steeper performance degradation (~21% accuracy) compared to gyroscope/magnetometer channels (~17-18%), indicating highly structure-dependent fault feature coupling.

### Robustness to Synchronization Errors & Data Quality

![fig4_robustness](figures/fig4_robustness.png)

**Figure 4. Model robustness under DT time synchronization error and data corruption.** Time synchronization error is simulated by modifying DT feature time alignment; DT data corruption is implemented by replacing DT channels with Gaussian noise. (a) Line plot of model accuracy under cross-mode DT data mixing. (b) Line plot under random time jitter ($\pm0$ to $\pm700$ sample offset). (c) Impact of resampling ratio ($0.5\sim2.0$) deviation on model accuracy. (d) Bar chart under DT data corruption ($1.0\times$ to $10.0\times$ noise level). (e) Confusion matrix of the full model on the test set. (f) Line plots of model Accuracy and F1 score under decreasing SNR ($40$ dB to $-10$ dB).

#### Key Findings

*   **Temporal Desynchronization Robustness:** The model handles severe synchronization errors remarkably well. It maintains >91% accuracy under $\pm100$ sample jitter or $\le50\%$ cross-modal mixing, and ~94% accuracy across $0.5\times$ to $2.0\times$ sampling rate deviations. This success stems from the fact that the graph structure relies primarily on **relational patterns** rather than strict temporal point-to-point numerical alignment.
*   **SNR and Data Corruption Boundaries:** A critical Signal-to-Noise Ratio (SNR) threshold exists at ~5 dB. When SNR drops below 0 dB or corruption exceeds the $1\sigma$ level, the diagnostic class structure completely collapses (accuracy approaching random guessing at ~15%).
*   **Role of the Digital Twin:** Once DT information is severely invalidated, the class structure collapses. This proves that the DT is not merely "supplementary information," but a critical reference for establishing decision boundaries. Our boundary tests conclusively demonstrate that the framework **does not require absolute high-fidelity non-linear modeling**, but relies strictly on **dynamic consistency and the absence of systematic offsets**.
