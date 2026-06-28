# Vehicle Gearbox Fault Diagnosis Using Physics-Informed AI & Deep Learning

## 📌 Project Overview
This repository contains an end-to-end Predictive Maintenance (PdM) framework developed using the **University of Connecticut (UConn) In-House Gearbox Dataset**. The project successfully isolates and diagnoses 9 distinct structural and mechanical conditions of a vehicle gearbox by analyzing high-frequency vibration signals. 

By bridging classical mechanical engineering domain expertise (**Digital Signal Processing**) with modern Artificial Intelligence, this framework evaluates two paradigms:
1. **Physics-Informed Feature Engineering** followed by a **Random Forest Classifier**.
2. **End-to-End Deep Learning** utilizing a **1-Dimensional Convolutional Neural Network (1D CNN)**.

Crucially, this project explicitly addresses and solves the critical industry issue of **Temporal Data Leakage** in continuous time-series benchmarks using strict **Block-Based Data Splitting**.

---

## 📊 Dataset & Engineering Specifications
* **Data Source:** UConn In-House Gearbox Vibration Dataset.
* **Sampling Frequency ($f_s$):** 20 kHz (20,000 samples/sec).
* **Sample Length ($N$):** 3,600 continuous acceleration data points per record (~0.18 seconds of structural operation).
* **Data Distribution:** 936 total records, perfectly balanced with 104 samples per condition.
* **Target Classes (9 Mechanical States):**
  * `H` - Healthy (Baseline Benchmark)
  * `MT` - Missing Tooth (Severe localized impact failure)
  * `RC` - Root Crack (Structural fatigue failure)
  * `SP` - Spalling (Surface roughness degradation)
  * `CH1` to `CH5` - Five progressive levels of gear tooth chipping defects

---

## 🛠️ Framework Architecture & Methodology

### 1. Digital Signal Processing (DSP) & Frequency Analysis
Before feeding data into traditional ML models, Fast Fourier Transform (FFT) is applied to transform raw acceleration arrays into the frequency domain. Averaging the FFT across contiguous segments allows the identification of distinct **Gear Mesh Frequencies (GMF)** and sidebands, reducing stochastic laboratory noise.

### 2. Physics-Informed Feature Extraction
To compress the 3,600 raw time steps into high-value mechanical indicators, 6 distinct statistical and frequency features were extracted for each sample:
* **Time-Domain:** Root Mean Square (RMS) for total energy tracking, Kurtosis and Crest Factor for transient shock/impulse detection, and Skewness for waveform asymmetry.
* **Frequency-Domain:** Mean Frequency to capture structural resonance and energy distribution shifts.

### 3. Mitigating Temporal Data Leakage (The Block Split)
Vibration data collected in continuous streams exhibits strong adjacent correlations. Using standard random shuffling (`train_test_split`) leaks neighboring temporal patterns into the test set, creating artificially inflated metrics (e.g., ~100% accuracy). 
* **Solution:** We implemented a strict **Sequential Block Split**, partitioning the first continuous 80% blocks for training and preserving the completely unseen terminal 20% blocks for evaluation. This accurately simulates real-world industrial deployments.

---

## 📈 Experimental Results & Performance

| Diagnostic Model | Splitting Strategy | Evaluation Accuracy | Engineering Insight |
| :--- | :--- | :--- | :--- |
| **Random Forest (Baseline)** | Random Shuffling | **98.40%** | Suffers from optimistic bias due to latent temporal data leakage. |
| **Random Forest (Realistic)** | **Strict Block Split** | **99.47%** | Solid generalization; successfully identifies fault categories using only 6 extracted features. |
| **1D CNN (Deep Learning)** | Continuous Raw Inputs | **100.00%** | Autonomous spatial feature extraction captures micro-shocks with absolute zero misclassifications. |

### Mechanical Key Takeaway:
The feature importance analysis highlighted **RMS** as the leading discriminator, heavily relying on its ability to smoothly track the progressive energy increases across the 5 levels of tooth chipping defects (`CH1`-`CH5`). 

---

## 💻 Tech Stack & Tools
* **Programming Language:** Python 3.12
* **Signal Processing & Math:** NumPy, SciPy (Signal Statistics)
* **Data Engineering:** Pandas
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn
* **Deep Learning Framework:** TensorFlow / Keras (1D CNN Architecture)

---

## 🚀 How to Run the Project
1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
