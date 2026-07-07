# Resilient Decentralized Edge Learning via Sentinel-Based Fault Localization and Containment

This repository contains the code accompanying the paper:

**“Resilient Decentralized Edge Learning via Sentinel-Based Fault Localization and Containment”**

accepted at **IDC 2026 — 18th International Symposium on Intelligent Distributed Computing**.

## Overview

This code implements a sentinel-based resilience framework for decentralized edge learning. The setting consists of multiple IoT/edge nodes, where each node trains a local LSTM predictor on its own time-series observations and exchanges only model parameters with neighboring nodes through gossip-based communication.

The framework studies the case where a hidden fault is injected into one unmonitored node. The corrupted model state can then propagate to neighboring nodes through parameter exchange. To limit this effect, a subset of monitored nodes, called **sentinels**, tracks local prediction residuals, raises alarms when abnormal behavior is detected, estimates the likely fault source using graph distances and alarm timings, and heals affected nodes using healthy neighbors.

The experimental case study uses real Received Signal Strength Indicator (RSSI) measurements from the CityLab testbed within the FED4FIRE+ federation.

## Parameter Reference

The main experiment parameters are defined near the beginning of `main - real data.ipynb`.

| Parameter | Value in code | Description |
|---|---:|---|
| `L` | `50` | Side length of the two-dimensional square deployment area. |
| `N` | `27` | Number of decentralized nodes. It is derived from the number of RSSI streams in the dataset. |
| `M` | `5` | Number of monitored sentinel nodes. |
| `M/N` | `5/27` | Sentinel-to-node ratio, controlling the degree of partial observability. |
| `margin_frac` | `0.05` | Margin used when generating node positions, preventing nodes from being placed too close to the deployment boundary. |
| `min_dist` | `5.0` | Minimum Euclidean distance allowed between generated node positions. |
| `r_max` | `L / 5` | Communication radius used to define graph neighbors. |
| `self_weight` | `0.05` | Weight assigned to a node’s own model parameters during gossip aggregation. |
| `hidden_idx` | generated | Indices of unmonitored nodes, excluding the sentinels. |
| `source_node` | generated | True hidden faulty node selected from the unmonitored nodes. |
| `seq_len` | `15` | Sliding-window length used by the LSTM predictor. |
| `B` | `100` | Number of RSSI samples processed per node at each communication round. |
| `R` | `min_len // B` | Number of communication rounds, computed from the shortest RSSI stream and the batch length. |
| `local_epochs` | `5` | Number of local training epochs performed by each node per communication round. |
| `input_dim` | `1` | Input dimension of the LSTM model; RSSI streams are univariate. |
| `hidden_dim` | `32` | Number of hidden units in the LSTM layer. |
| `layer_dim` | `1` | Number of stacked LSTM layers. |
| `output_dim` | `1` | Output dimension of the LSTM model; one future RSSI value is predicted. |
| `learning_rate` | `0.001` | Learning rate used by the Adam optimizer. |
| `optimizer` | `Adam` | Optimizer used for local LSTM training. |
| `drift_round` | `25` | Communication round at which the hidden fault is introduced. |
| `fault_round` | `drift_round` | Alias for the fault-injection round. |
| `warmup_rounds` | `20` | Number of initial fault-free rounds used to establish nominal sentinel residual behavior. |
| `fault_weight_shift` | `5.0` | Additive perturbation applied to the output-layer weights of the faulty node. |
| `fault_bias_shift` | `2.5` | Additive perturbation applied to the output-layer bias of the faulty node. |
| `mode` | `"all"` | Applies the output-layer perturbation to all relevant output parameters. |
| `rho` | `0.05` | Smoothing factor used when updating the baseline residual energy. |
| `tau_det` | `2` | Timing tolerance used during source and affected-region estimation. |
| `guard_hops` | `1` | Additional hop margin added around the estimated affected region. |
| `min_required_alarms` | `2` | Minimum number of sentinel alarms required before source localization is attempted. |



## Repository Contents

```text
.
├── main - real data.ipynb      # Main experiment notebook
├── utils.py                    # Utility functions used by the notebook
├── rssi_cl.xlsx                # RSSI dataset file
└── README.md                   # Repository documentation
