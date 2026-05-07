# Appendix: CollectiveOS-Bench

## A. Full Per-Seed Results

### A.1 Commons Environment — All Seeds

| Baseline | Seed | CR | SS | Gini | CEI |
|---|---|---|---|---|---|
| PPO | 42 | 0.73 | 0.83 | 0.18 | 0.75 |
| PPO | 123 | 0.69 | 0.79 | 0.21 | 0.70 |
| PPO | 2024 | 0.74 | 0.84 | 0.17 | 0.76 |
| PPO | 7 | 0.68 | 0.80 | 0.22 | 0.71 |
| PPO | 999 | 0.71 | 0.84 | 0.16 | 0.73 |
| **PPO mean ± std** | — | **0.71 ± 0.06** | **0.82 ± 0.05** | **0.19 ± 0.04** | **0.73 ± 0.05** |
| Cooperative | 42 | 0.92 | 0.95 | 0.06 | 0.90 |
| Greedy | 42 | 0.10 | 0.18 | 0.53 | 0.07 |
| Random | 42 | 0.51 | 0.44 | 0.32 | 0.33 |

### A.2 Institution Environment — All Seeds

| Baseline | Seed | CR | IES | CAR | CEI |
|---|---|---|---|---|---|
| PPO | 42 | 0.74 | 0.67 | 0.55 | 0.76 |
| PPO | 123 | 0.70 | 0.62 | 0.50 | 0.72 |
| PPO | 2024 | 0.75 | 0.68 | 0.57 | 0.77 |
| PPO | 7 | 0.68 | 0.61 | 0.49 | 0.70 |
| PPO | 999 | 0.73 | 0.67 | 0.54 | 0.75 |
| **PPO mean ± std** | — | **0.72 ± 0.07** | **0.65 ± 0.07** | **0.53 ± 0.08** | **0.74 ± 0.05** |

### A.3 Market Environment — All Seeds

| Baseline | Seed | SW (norm.) | Mkt. Eff. | Gini | CEI |
|---|---|---|---|---|---|
| PPO | 42 | 0.70 | 0.73 | 0.20 | 0.68 |
| PPO | 123 | 0.66 | 0.69 | 0.24 | 0.63 |
| PPO | 2024 | 0.71 | 0.74 | 0.19 | 0.69 |
| PPO | 7 | 0.64 | 0.67 | 0.25 | 0.61 |
| PPO | 999 | 0.69 | 0.72 | 0.22 | 0.67 |
| **PPO mean ± std** | — | **0.68 ± 0.07** | **0.71 ± 0.06** | **0.22 ± 0.05** | **0.66 ± 0.06** |

---

## B. Hyperparameter Sensitivity

### B.1 PPO Hyperparameters (Institution Environment, seed=42)

| lr | clip_eps | n_epochs | hidden | IES | CEI |
|---|---|---|---|---|---|
| 1e-4 | 0.2 | 4 | 128 | 0.58 | 0.66 |
| **3e-4** | **0.2** | **4** | **128** | **0.67** | **0.76** |
| 1e-3 | 0.2 | 4 | 128 | 0.51 | 0.59 |
| 3e-4 | 0.1 | 4 | 128 | 0.63 | 0.72 |
| 3e-4 | 0.3 | 4 | 128 | 0.65 | 0.73 |
| 3e-4 | 0.2 | 2 | 128 | 0.60 | 0.68 |
| 3e-4 | 0.2 | 8 | 128 | 0.64 | 0.73 |
| 3e-4 | 0.2 | 4 | 64 | 0.59 | 0.67 |
| 3e-4 | 0.2 | 4 | 256 | 0.66 | 0.75 |

Bold = default configuration. Results show robustness to standard PPO hyperparameter ranges.

### B.2 Environment Parameter Sensitivity (Institution, PPO)

| enforcement (ε) | sanction (σ) | IES | CR | CEI |
|---|---|---|---|---|
| 0.2 | 0.3 | 0.42 | 0.51 | 0.48 |
| 0.5 | 0.3 | 0.59 | 0.65 | 0.65 |
| **0.7** | **0.3** | **0.65** | **0.72** | **0.74** |
| 0.9 | 0.3 | 0.68 | 0.76 | 0.77 |
| 0.7 | 0.1 | 0.55 | 0.63 | 0.61 |
| 0.7 | 0.5 | 0.67 | 0.74 | 0.76 |
| 0.7 | 0.8 | 0.64 | 0.70 | 0.72 |

Finding: Credible enforcement (ε ≥ 0.7) is necessary for IES ≥ 0.6, consistent with Ostrom's design principle of effective monitoring.

---

## C. Metric Derivations

### C.1 IES Decomposition

The Institutional Emergence Score decomposes into three factors:

```
IES = A × C × S

A = adaptation rate   ∈ [0,1]   "Did the institution evolve?"
C = compliance rate   ∈ [0,1]   "Did agents follow it?"
S = survival          ∈ {0,1}   "Did it prevent collapse?"
```

This multiplicative form ensures that any single failure — no adaptation, no compliance, or collapse — drives IES to zero. An institution that adapted well but collapsed (S=0) scores 0; an institution that maintained compliance but never adapted (A≈0) also scores near 0.

**Why multiplicative rather than additive?**  
An additive form (IES = (A + C + S)/3) would give partial credit for institutions that collapse (S=0) as long as A and C are high. But a collapsed commons represents total institutional failure regardless of how frequently rules were updated.

### C.2 CEI Normalisation

The Coordination Efficiency Index borrows from mechanism design efficiency theory:

```
CEI = (SW_actual - SW_Nash) / (SW_optimal - SW_Nash)
```

**SW_Nash** is estimated by running the Greedy baseline (always-defect) for 20 episodes and taking the mean social welfare. This approximates the non-cooperative equilibrium where no agent restricts its extraction.

**SW_optimal** is estimated using the `CentralizedOptimalController` analytic solution: all agents extract 1/N of 90% of the sustainable yield, maximising long-run resource productivity.

Both estimates are environment-specific and recomputed when environment parameters change. Default estimates (used in paper experiments):
- Commons: SW_Nash ≈ 850, SW_optimal ≈ 4200 (over 200 steps, 5 agents)
- Institution: SW_Nash ≈ 600, SW_optimal ≈ 5100 (over 300 steps, 6 agents)

### C.3 CAR Interpretation

```
CAR = (1/|Δ|) · Σ_k |Δθ_k| · v_k
```

**Example:** In one episode, two rule changes occurred:
- Change 1: threshold 0.5→0.7 with 80% vote (score: 0.2 × 0.8 = 0.16)
- Change 2: threshold 0.7→0.65 with 55% vote (score: 0.05 × 0.55 = 0.028)

CAR = (0.16 + 0.028) / 2 = 0.094

This penalises both small changes (low |Δθ|) and low-legitimacy changes (low v_k). A single large, unanimously-supported rule change scores higher than many small, barely-passed amendments.

---

## D. Environment Design Rationale

### D.1 Why Logistic Growth?

The logistic model R_{t+1} = R_t + r·R·(1-R/K) has well-known properties:
- Sustainable yield is maximised at R = K/2 with yield rK/4
- Below the collapse threshold, regeneration cannot compensate extraction
- Parameters r and K have clear ecological interpretations

This provides a theoretically grounded commons that behaves consistently across parameter regimes.

### D.2 Why Majority Voting for Rule Proposals?

Majority voting (50% threshold) reflects the minimal democratic institution. Future work could parameterise the voting threshold (e.g., supermajority for large threshold changes) as a governance design variable. The current design gives agents a meaningful mechanism for institutional change without requiring complex commitment devices.

### D.3 Why Two Goods in the Market?

Two goods with heterogeneous endowments create gains from trade — the condition under which markets are non-trivially useful. With one good or homogeneous endowments, rational agents have no reason to trade. The two-good design forces agents to discover the mutual benefit of specialisation and exchange.

---

## E. Comparison to Existing Benchmarks

| Feature | SMAC | MPE | Melting Pot | **CollectiveOS-Bench** |
|---|---|---|---|---|
| Dynamic rule systems | ✗ | ✗ | ✗ | **✓** |
| Agent-authored governance | ✗ | ✗ | ✗ | **✓** |
| IES / CAR / CEI metrics | ✗ | ✗ | ✗ | **✓** |
| Renewable resource dynamics | ✗ | ✗ | Partial | **✓** |
| Market with price discovery | ✗ | ✗ | ✗ | **✓** |
| Config-driven reproducibility | Partial | ✗ | Partial | **✓** |
| REST API | ✗ | ✗ | ✗ | **✓** |
| PyTorch native agents | ✗ | ✗ | ✗ | **✓** |
| Open-source (Apache 2.0) | ✓ | ✓ | ✓ | **✓** |

---

## F. Limitations and Future Work

1. **Rule space dimensionality.** The current Institution environment parameterises rules as a single scalar (extraction threshold). Real institutions involve multi-dimensional rule spaces. Future work should extend to vector-valued rules covering allocation, exclusion, sanctioning gradient, and monitoring intensity.

2. **Population generalisation.** All experiments use fixed agent populations. Evaluating generalisation to held-out agent populations (as in Melting Pot) is an important extension.

3. **Partial observability.** Agents currently observe the global resource level. Adding observation noise or masking resource information would test whether institutional mechanisms compensate for informational asymmetries.

4. **Non-stationarity.** Environmental parameters (r, K) are fixed per episode. Introducing slow parameter drift would test whether institutions can adapt to changing ecological conditions — a key challenge in real commons governance.

5. **Richer communication.** The CommNet baseline uses a minimal broadcast channel. Structured communication protocols, commitment devices, and cheap-talk mechanisms are natural extensions.

6. **Human-in-the-loop evaluation.** Testing whether human participants can improve collective outcomes when paired with AI agents in the Institution environment is a compelling future direction.
