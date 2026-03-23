# Dataset: V2V c4 (Training & In-Domain Test)

This directory contains the V2V communication mode dataset for the c4 (command set 4) flight test campaign.

---

## Directory Structure

```
data_for_v2v_c4/
├── Draw/              # Raw sensor recordings from physical UAV
├── Dsim/              # Digital twin simulation outputs (CopterSim)
├── Dpro/              # Preprocessed & synchronized data (ready for training)
│   ├── 1-hover-normal-v2v/
│   ├── 2-forward-normal-v2v/
│   ├── 3-acc-x-axis-flag3-3000-normal-v2v/
│   ├── 5-motor03-flag4-085-normal-v2v/
│   └── 6-motor03-flag3-200-normal-v2v/
└── Dpro.py             # Preprocessing script: Draw + Dsim → Dpro
```

---

## Preprocessed Data Files (`Dpro/`)

Each scenario folder in `Dpro/` contains synchronized CSV triplets:

| File Pattern | Content | Description |
|-------------|---------|-------------|
| `all_sycn_raw_data_mode{phase}.csv` | Physical sensor readings | Raw IMU data from real UAV |
| `all_sycn_sim_data_mode{phase}.csv` | DT simulation output | CopterSim synchronized output |
| `all_sycn_err_data_mode{phase}.csv` | Residual error | `err = raw − sim` |

### Scenario Details

| Scenario | Directory | Class | Phase | Rows (samples) |
|----------|-----------|:-----:|:-----:|-------:|
| Normal (Hover) | `1-hover-normal-v2v` | C1 | `2_6` | 3,520 |
| Normal (Forward) | `2-forward-normal-v2v` | C1 | `2_5` | 2,709 |
| Acc-X Sensor Fault | `3-acc-x-axis-flag3-3000-normal-v2v` | C2 | `2_9` | 1,616 |
| Motor Disturbance | `6-motor03-flag3-200-normal-v2v` | C3 | `2_9` | 1,654 |
| Efficiency Degradation | `5-motor03-flag4-085-normal-v2v` | C4 | `2_9` | 847 |

**Phase encoding**: `mode{X}_{Y}` where X = flight stage, Y = sub-phase. Phase `2_9` indicates fault-active state.

---

## Raw CSV Column Layout

Each CSV file has **9 columns** (no header row in some versions):

| Column Index | Sensor | Channel | Unit |
|:---:|--------|---------|------|
| 0 | Gyroscope | gyro_rad[0] (X-axis) | rad/s |
| 1 | Gyroscope | gyro_rad[1] (Y-axis) | rad/s |
| 2 | Gyroscope | gyro_rad[2] (Z-axis) | rad/s |
| 3 | Accelerometer | accelerometer_m_s2[0] (X-axis) | m/s² |
| 4 | Accelerometer | accelerometer_m_s2[1] (Y-axis) | m/s² |
| 5 | Accelerometer | accelerometer_m_s2[2] (Z-axis) | m/s² |
| 6 | Magnetometer | magnetometer_ga[0] (X-axis) | Gauss |
| 7 | Magnetometer | magnetometer_ga[1] (Y-axis) | Gauss |
| 8 | Magnetometer | magnetometer_ga[2] (Z-axis) | Gauss |

**Sampling rate**: 50 Hz (20 ms per sample)

---

## Data Preprocessing Pipeline

### Step 1: Channel Reordering

Raw CSV data (Gyro, Acc, Mag order) is reordered to **Acc-first** layout:

```python
# Original CSV: [Gyro(3), Acc(3), Mag(3)]
rs = np.split(raw, 3, axis=1)  # rs[0]=Gyro, rs[1]=Acc, rs[2]=Mag
r_ = np.hstack((rs[1], rs[0], rs[2]))  # → [Acc(3), Gyro(3), Mag(3)]
```

### Step 2: Raw-Error Interleaving (18-Channel Construction)

Raw and DT-error data are interleaved by sensor group:

```python
# r2[0]=Acc(3), r2[1]=Gyro(3), r2[2]=Mag(3)
# e2[0]=ErrAcc(3), e2[1]=ErrGyro(3), e2[2]=ErrMag(3)
raw_err = np.hstack((r2[0], e2[0], r2[1], e2[1], r2[2], e2[2]))
```

**Final 18-channel layout:**

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|-------|---|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|----|----|
| **Content** | Acc-x | Acc-y | Acc-z | Err-Acc-x | Err-Acc-y | Err-Acc-z | Gyro-x | Gyro-y | Gyro-z | Err-Gyro-x | Err-Gyro-y | Err-Gyro-z | Mag-x | Mag-y | Mag-z | Err-Mag-x | Err-Mag-y | Err-Mag-z |

### Step 3: Sliding Window Segmentation

```python
SEQUENCE_LENGTH = 80   # Timesteps per sample (80 × 0.02s = 1.6s window)
STEP_SIZE = 4          # Stride between windows (4 × 0.02s = 0.08s)

# Produces sequences of shape (N, 80, 18)
seqs = [data[i:i+80] for i in range(0, len(data)-80+1, 4)]
```

### Step 4: Graph Construction

Each sample `(80, 18)` is reshaped into a spatio-temporal graph:

- **6 sensor nodes** per timestep: `[Acc, Err-Acc, Gyro, Err-Gyro, Mag, Err-Mag]`
- Each node has **3 features** (x, y, z components)
- Total: `6 × 80 = 480` nodes per sample

**Edge types** (6 types):

| Edge Type | ID | Connection | Weight | Description |
|-----------|:--:|-----------|:------:|-------------|
| Temporal | 0 | $s_t → s_{t+1}$ | 0.1 | Same sensor across time |
| Spatial-Cross | 1–4 | Between Acc/Gyro nodes | 0.2 | Cross-sensor spatial coupling |
| Raw-Error | 5 | $\text{raw}_t ↔ \text{err}_t$ | 0.7 | Raw-to-DT-error pairing |

```python
# Edge weight modulation in MyGATConv:
alpha_w[edge_type == 0] = 0.1    # Temporal: low weight
alpha_w[edge_type ∈ {1,2,3,4}] = 0.2  # Spatial cross-coupling
alpha_w[edge_type == 5] = 0.7    # Raw↔Error: highest weight
```

---

## Usage Example

```python
import pandas as pd
import numpy as np

# Load a scenario
raw = pd.read_csv('Dpro/1-hover-normal-v2v/all_sycn_raw_data_mode2_6.csv').to_numpy()
err = pd.read_csv('Dpro/1-hover-normal-v2v/all_sycn_err_data_mode2_6.csv').to_numpy()

# Step 1: Reorder channels (Gyro,Acc,Mag → Acc,Gyro,Mag)
rs, es = np.split(raw, 3, axis=1), np.split(err, 3, axis=1)
r_ = np.hstack((rs[1], rs[0], rs[2]))
e_ = np.hstack((es[1], es[0], es[2]))

# Step 2: Interleave raw and error
r2 = [r_[:,0:3], r_[:,3:6], r_[:,6:9]]
e2 = [e_[:,0:3], e_[:,3:6], e_[:,6:9]]
data_18ch = np.hstack((r2[0], e2[0], r2[1], e2[1], r2[2], e2[2]))
# data_18ch.shape = (3520, 18)

# Step 3: Sliding window
SEQ_LEN, STEP = 80, 4
seqs = np.array([data_18ch[i:i+SEQ_LEN] for i in range(0, len(data_18ch)-SEQ_LEN+1, STEP)])
# seqs.shape = (N, 80, 18)
```
