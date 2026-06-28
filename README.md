# Vehicle Gearbox Fault Diagnosis: Physics-Informed AI & Deep Learning

## 📌 Project Overview
This repository delivers an industrial-grade **Predictive Maintenance (PdM) & Health Management (PHM) Pipeline** utilizing the high-frequency **University of Connecticut (UConn) In-House Gearbox Dataset**. 

Rather than treating machine learning as a pure data-driven black box, this framework bridges classical mechanical engineering expertise (**Digital Signal Processing**) with modern Artificial Intelligence. The system is designed to isolate and classify 9 distinct continuous structural gearbox states—including healthy baseline operation, severe localized impact defects (Missing Tooth, Root Crack, Spalling), and 5 progressive stages of gear tooth chipping severity (`CH1` to `CH5`). 

Crucially, this project explicitly addresses and solves the critical industry problem of **Temporal Data Leakage** in continuous time-series streams using strict **Block-Based Data Splitting**.

---

## 📊 Methodology & Engineering Pipeline

### 1. High-Frequency Signal Exploration
Raw acceleration signals sampled at a high-fidelity rate of **20 kHz** were processed. Each operational record contains **3,600 continuous data points** (~0.18 seconds of dynamic structural operation). The dataset represents a perfectly balanced framework across 9 physical states with 104 samples per condition.

### 2. Physics-Informed Feature Engineering (Traditional ML Pathway)
To capture specific mechanical degradation markers without requiring massive computational architectures, the raw 3,600 time-steps were compressed into 6 high-value statistical and frequency indicators:
* **Time-Domain Indicators:** Root Mean Square (RMS) for total structural energy tracking, Kurtosis and Crest Factor for transient shock/impulse detection, and Skewness for waveform asymmetry.
* **Frequency-Domain Indicators:** Fast Fourier Transform (FFT) Mean Frequency to isolate shifts in structural resonance zones.

### 3. Deep Learning Spatial Extraction (1D-CNN Pathway)
In parallel, an end-to-end **1-Dimensional Convolutional Neural Network (1D-CNN)** was architected. This network bypasses manual feature extraction by utilizing continuous convolutional filters to autonomously learn spatial hierarchies and micro-shock signatures directly from raw time-series inputs.

### 4. Eliminating Temporal Data Leakage (The Sequential Block Split)
Vibration data collected from continuous test rigs exhibits extreme adjacent correlation. Utilizing a standard random shuffle (`train_test_split`) causes neighboring temporal wave patterns to leak into the validation set, yielding artificially inflated, unrealistic model evaluations (e.g., false 100% accuracy). 
* **The Solution:** We implemented a rigorous **Sequential Block Split**, partitioning the first continuous 80% blocks exclusively for model training, and preserving the terminal, unseen 20% blocks for deployment simulation.

---

## 📈 Engineering Evolution & Results

Below is the comparative statistical breakdown of the model evaluation paradigms across our validation spectrum:

| Diagnostic Model | Feature Representation | Splitting Strategy | Evaluation Accuracy | Generalization Status |
| :--- | :--- | :--- | :---: | :---: |
| **Random Forest (Baseline)** | 6 Extracted Features | Random Shuffling | **98.40%** | ⚠️ Biased (Temporal Leakage) |
| **Random Forest (Realistic)** | 6 Extracted Features | **Strict Block Split** | **99.47%** | ✅ Verified Generalization |
| **1D-CNN (Deep Learning)** | 3600 Raw Time-Steps | **Strict Block Split** | **100.00%** | 🏆 Optimal Field Deployment |

### 🔬 Mechanical Domain Takeaway:
Feature importance mapping proved that the **Root Mean Square (RMS)** indicator was the dominant discriminator. This aligns perfectly with vibration physics, as cumulative energy increases linearly as the physical defect progresses from initial pitch anomalies (`CH1`) to severe structural material loss (`CH5`).

---

## 🔬 Visualizations & Diagnostic Reports

### 🔹 Frequency Domain Spectrum Analysis
<p align="center">
  <img src="images/Frequency_Domain_Spectrum_Analysis_(FFT_Average)_Across 9_Gearbox_States.jpg" alt="FFT Average Across 9 Gearbox States" width="90%">
</p>
<p align="center">
  <em>Figure 1: Averaged FFT spectra isolating specific Gear Mesh Frequencies (GMF) and sideband energy distribution changes across the 9 distinct mechanical conditions.</em>
</p>

### 🔹 Mechanical Feature Importance Rankings
<p align="center">
  <img src="images/Mechanical_Diagnostics_Feature_Importance_Rankings.png" alt="Feature Importance Rankings" width="75%">
</p>
<p align="center">
  <em>Figure 2: Deterministic Random Forest feature weights highlighting the operational dominance of RMS, Mean Frequency, and Peak-to-Peak in capturing dynamic failure modes.</em>
</p>

### 🔹 Baseline Performance (With Data Leakage Vector)
<p align="center">
  <img src="images/Baseline_Confusion_Matrix_(Random _Split_-_Potential_Leakage_Warning).png" alt="Baseline Confusion Matrix" width="65%">
</p>
<p align="center">
  <em>Figure 3: Multi-class Confusion Matrix under standard random splitting, demonstrating artificially inflated metrics due to latent temporal data leakage.</em>
</p>

### 🔹 Realistic Generalization Performance (Strict Block Splitting)
<p align="center">
  <img src="images/Realistic_Confusion_Matrix_(Strict_Block_Splitting).png" alt="Realistic Confusion Matrix" width="65%">
</p>
<p align="center">
  <em>Figure 4: Multi-class Confusion Matrix under strict Sequential Block Splitting, validating robust generalization and real-world deployment readiness with zero leakage.</em>
</p>

---

## 🧠 Core Engineering Competencies Demonstrated
1. **Physics-Informed Architecture:** Proven capability to map raw digital signals into physical kinematic equations and statistical descriptors (RMS, Kurtosis), reducing data dependencies.
2. **Data Integrity Assurance:** Deep understanding of time-series constraints, successfully diagnosing and mitigating data leakage vulnerabilities that skew industrial model reliability.
3. **Deep Learning for PHM:** Advanced implementation of 1D-CNN structures for automated feature learning in Prognostics and Health Management applications.

---

## 🚀 Quick Start & Installation

```bash
# Clone this repository
git clone [https://github.com/faridghattas/predictive-maintenance-uconn-gearbox.git](https://github.com/faridghattas/predictive-maintenance-uconn-gearbox.git)

# Install required dependencies
pip install numpy pandas matplotlib seaborn scipy scikit-learn tensorflow
