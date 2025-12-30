## CATCH-RG: Channel-Aware Multivariate Time-Series Anomaly Detection via Patching with Reliability-Gated Channel Mask Generator

### Recognition

This work received an **Excellence Award at KAMP 2025** (*The 5th K-Artificial Intelligence Manufacturing Data Analysis Competition*),  
recognizing its robustness-oriented extension of **CATCH : Channel-Aware multivariate Time Series Anomaly Detection via Frequency Patching (ICLR'25)** for real-world anomaly detection scenarios involving imperfect and incomplete multivariate time-series data.

---

### Overview

**CATCH** is a channel-aware multivariate time-series anomaly detection method that models cross-channel interactions through frequency-domain patching and channel-aware masking. It is evaluated under the **TAB** benchmarking protocol, which provides unified and rigorous evaluation for time-series anomaly detection. This repository provides a **robust masking extension** for **CATCH**, and proposed **reliability-gated channel masking** to better handle **incomplete and noisy multivariate time-series data**, while preserving the original CATCH architecture and interface.

---

### Motivation

In practical settings, multivariate time-series data often contain:
- missing or partially observed channels,
- noisy or corrupted sensor measurements,
- globally inactive segments due to sensor or communication failures.

The original CATCH framework does not explicitly model channel reliability, which may lead to unstable cross-channel interactions under such conditions.

---

### Reliability-Gated Mask (RG-Mask)

We extend the channel masking mechanism of CATCH with a **Reliability-Gated Mask**, which dynamically controls cross-channel information flow based on data-driven channel reliability estimates.

The proposed mask:
- estimates channel reliability using simple, robust statistics,
- softly suppresses interactions involving unreliable channels,
- preserves self-channel connections,
- and collapses to an identity mask when all channels are globally inactive.

---

## Citation

If you use this work or build upon it, please cite the original CATCH and TAB papers:

```bibtex
@inproceedings{wu2024catch,
  title     = {{CATCH}: Channel-Aware multivariate Time Series Anomaly Detection via Frequency Patching},
  author    = {Wu, Xingjian and Qiu, Xiangfei and Li, Zhengyu and Wang, Yihang and Hu, Jilin and Guo, Chenjuan and Xiong, Hui and Yang, Bin},
  booktitle = {ICLR},
  year      = {2025}
}
