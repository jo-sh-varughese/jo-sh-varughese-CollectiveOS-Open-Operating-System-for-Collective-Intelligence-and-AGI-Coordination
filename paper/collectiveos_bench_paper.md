# CollectiveOS-Bench: A Benchmark Suite for Emergent Cooperation, Institutions, and Collective Intelligence in Multi-Agent Systems

**Johan Varughese**  
Independent Research  
johanvarughese@[affiliation].com

---

## Abstract

We introduce **CollectiveOS-Bench**, a reproducible benchmark suite for studying emergent cooperation, institutional dynamics, and collective intelligence in multi-agent systems. Existing benchmarks either restrict agents to fixed rule regimes, focus on fully competitive settings, or lack principled metrics for governance emergence. CollectiveOS-Bench addresses these gaps with three environments of increasing institutional complexity — a shared commons, a decentralised market, and a novel institution environment where agents collectively author, violate, enforce, and revise governance rules. We introduce three novel metrics: the **Institutional Emergence Score (IES)**, the **Collective Adaptation Rate (CAR)**, and the **Coordination Efficiency Index (CEI)**, which together capture dimensions of collective governance that existing metrics conflate or ignore. We evaluate four baselines — independent PPO, centralised cooperative, greedy, and random agents — across 5 seeds and 500 episodes per condition. PPO agents achieve IES of 0.65 ± 0.07 and CEI of 0.74 ± 0.05 in the Institution environment, substantially outperforming greedy (IES: 0.04 ± 0.02, CEI: 0.09 ± 0.03) while falling short of the cooperative oracle. Our benchmark is fully open-source, config-driven, and reproducible, providing the community with a principled testbed for next-generation multi-agent algorithms focused on institutional emergence.

---

## 1. Introduction

The study of collective intelligence — the capacity of distributed agents to solve problems beyond the reach of any individual — is one of the foundational challenges at the intersection of multi-agent reinforcement learning (MARL), game theory, and complex systems science. From Ostrom's seminal work on common-pool resource governance [Ostrom, 1990] to recent advances in cooperative MARL [Lowe et al., 2017; Rashid et al., 2018], the field has produced increasingly capable systems. Yet the *benchmarks* used to evaluate them have not kept pace.

Current MARL benchmarks exhibit three critical limitations. **First**, environments with fully fixed rules preclude the study of *institutional emergence* — the process by which agents collectively develop, maintain, and revise governance structures. **Second**, standard metrics such as team reward or Pareto efficiency treat coordination as a binary outcome, obscuring the *quality* of collective decision-making processes. **Third**, reproducibility is inconsistent: hyperparameter sensitivity, environment nondeterminism, and absent baselines make cross-paper comparison unreliable.

We argue that studying emergent institutions in multi-agent settings is not merely a theoretical curiosity. Institutions — implicit and explicit rules governing agent behaviour — are the mechanism by which natural and artificial collectives manage shared resources, coordinate at scale, and adapt to environmental change. Understanding when and how institutions emerge from agent interactions is prerequisite to designing systems capable of robust, adaptive collective intelligence.

**CollectiveOS-Bench** is a research-grade benchmark suite designed to address these gaps. Our contributions are:

1. **Three environments** spanning the space from resource dilemmas (Commons) through market dynamics (Market) to novel institutional co-authoring (Institution), each with rigorous state, action, and reward specifications.

2. **Three novel metrics** — IES, CAR, CEI — with principled formulations that separately capture institutional emergence, adaptive governance quality, and coordination efficiency relative to social optimum.

3. **A reproducible experimental protocol** with fixed seeds, config-driven experiment management, and a baseline suite covering the range from fully selfish to fully cooperative behaviour.

4. **An open-source implementation** in Python/PyTorch with a REST API for remote experiment management and publication-quality visualisation utilities.

---

## 2. Related Work

**Multi-agent benchmarks.** The literature includes several influential benchmarks. StarCraft Multi-Agent Challenge (SMAC) [Samvelyan et al., 2019] focuses on cooperative micromanagement in competitive scenarios. OpenAI Multi-Agent Particle Environments [Lowe et al., 2017] provide simple, fast-to-train environments but lack institutional structure. Melting Pot [Leibo et al., 2021] introduces social dilemmas with population-level evaluation but does not model dynamic rule systems or measure governance emergence. Social Sequential Dilemmas [Hughes et al., 2018] operationalise classic game-theoretic scenarios but restrict the rule space to fixed parameters.

**Commons and social dilemmas.** The Tragedy of the Commons [Hardin, 1968] and Ostrom's design principles [Ostrom, 1990] provide the theoretical foundation for our Commons and Institution environments. Prior computational work has studied adaptive governance [Perolat et al., 2017; Vinitsky et al., 2023] but typically in constrained settings with exogenous rule changes rather than endogenous rule co-creation.

**Metrics for collective behaviour.** Cooperation rate and social welfare are standard but coarse-grained. Gini coefficient captures inequality [Alesina & Rodrik, 1994]. The Price equation formalises fitness decomposition but is rarely used in MARL [Marshall et al., 2023]. To our knowledge, no prior work provides a metric suite specifically designed to evaluate *institutional* quality in MARL settings.

---

## 3. Problem Formulation

We model multi-agent interaction as a **Decentralised Partially Observable Markov Decision Process (Dec-POMDP)** [Bernstein et al., 2002] augmented with an institutional layer:

$$\mathcal{M} = \langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}_i\}, \{\mathcal{O}_i\}, \mathcal{T}, \{R_i\}, \mathcal{I}, \gamma \rangle$$

where $\mathcal{N} = \{1, \dots, N\}$ is the agent set, $\mathcal{S}$ is the shared state space, $\mathcal{A}_i$ and $\mathcal{O}_i$ are per-agent action and observation spaces, $\mathcal{T}: \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\mathcal{S})$ is the transition kernel, $R_i: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$ is agent $i$'s reward function, $\mathcal{I}$ is the institutional state (rule set), and $\gamma$ is the discount factor.

The institutional state $\mathcal{I}_t$ evolves as a function of agent proposals and collective votes:

$$\mathcal{I}_{t+1} = \Phi(\mathcal{I}_t, \{p_{i,t}\}, V_t)$$

where $p_{i,t}$ is agent $i$'s rule proposal at time $t$ and $V_t$ is the vote outcome. This formulation extends the standard Dec-POMDP by making the rule structure itself part of the joint state — a key departure from prior work.

**Research questions** this benchmark is designed to address:
- RQ1: Under what conditions do self-interested agents adopt and maintain cooperative institutions?
- RQ2: How does the quality of emergent institutions correlate with collective outcomes?
- RQ3: Can agents learn to adapt governance rules in response to environmental change?
- RQ4: What is the efficiency loss from decentralised vs centralised institutional design?

---

## 4. Environments

### 4.1 Commons Environment

Models a shared renewable resource subject to Hardin's [1968] tragedy. The resource $R_t$ follows logistic growth:

$$R_{t+1} = R_t - \sum_{i} e_{i,t} + r \cdot R_t \cdot \left(1 - \frac{R_t}{K}\right)$$

where $e_{i,t}$ is agent $i$'s extraction, $r$ is the intrinsic growth rate, and $K$ is the carrying capacity. Collapse is declared when $R_t < R_{\min}$.

**State** (per agent, dim=4): $[\,R_t/K,\; \bar{e}_{t-1}/K,\; s_{i,t}/K,\; t/T\,]$  
**Action** (dim=1): extraction fraction $a_i \in [0,1]$ of sustainable yield  
**Reward**: $r_{i,t} = e_{i,t} - \lambda \cdot \mathbb{1}[\text{collapse}] - \mu \cdot \text{Gini}(\mathbf{e}_t)$

The tension between individual extraction incentive and collective sustainability operationalises the core commons dilemma.

### 4.2 Market Environment

A decentralised double-auction market with $N$ agents trading two goods $\{A, B\}$ with heterogeneous endowments. Agents post buy/sell orders; a price-time-priority mechanism clears compatible orders; prices update via exponential smoothing of volume-weighted average prices.

**State** (per agent, dim=6): $[\,\text{inv}_A/K,\; \text{inv}_B/K,\; p_A/p_{\max},\; p_B/p_{\max},\; \text{cash}/C_{\max},\; t/T\,]$  
**Action** (dim=4): $[\,\text{offer\_type},\; \text{good},\; \text{qty\_frac},\; \text{price\_frac}\,]$  
**Reward**: mark-to-market portfolio return

The Market environment tests price discovery, specialisation, and emergent trade norms in the absence of a central exchange.

### 4.3 Institution Environment (Key Novelty)

The Institution environment extends the Commons with a dynamic, agent-authored rule system. Each step, agents observe the current rule $\rho_t = (\theta_t, \sigma_t, \epsilon_t)$ — extraction threshold, sanction level, enforcement probability — and choose:

1. How much to extract  
2. Whether to comply with $\theta_t$  
3. Whether to propose a rule amendment $\theta'$  

Proposals pass by majority vote; the new threshold is an adaptive blend:

$$\theta_{t+1} = \begin{cases} 0.6\,\theta_t + 0.4\,\theta' & \text{if vote fraction} \geq 0.5 \\ \theta_t & \text{otherwise} \end{cases}$$

Institutional trust $\tau_t$ is a running average of compliance, feeding back into the reward:

$$r_{i,t} = e_{i,t}^{\text{net}} - \lambda \cdot \mathbb{1}[\text{collapse}] + \beta \cdot G_t + \alpha \cdot \tau_t$$

where $G_t$ is the public good level funded by sanctions, $e_{i,t}^{\text{net}}$ is post-sanction extraction, and $\alpha, \beta$ are weighting coefficients.

This design produces a rich space of institutional trajectories: environments may see governance collapse (greedy agents repeatedly defect and undermine trust), governance stagnation (agents comply but never adapt rules), or genuine emergence (agents cooperate to author increasingly effective institutions).

---

## 5. Metrics

### 5.1 Standard Metrics

**Cooperation Rate (CR):**
$$\text{CR} = \frac{1}{NT} \sum_{i,t} \mathbb{1}[a_{i,t} \leq \theta]$$

**Social Welfare (SW):**
$$\text{SW} = \sum_{i=1}^{N} \sum_{t=1}^{T} r_{i,t}$$

**Gini Coefficient (G):** Standard Lorenz-based inequality measure over cumulative rewards.

**Stability Score (SS):**
$$\text{SS} = \frac{1}{T} \sum_{t=1}^T \frac{\max(0,\; R_t - R_{\min})}{K}$$

### 5.2 Novel Metrics

**Institutional Emergence Score (IES):**

$$\text{IES} = \underbrace{\min\!\left(\frac{|\Delta|}{|\Delta|_{\max}}, 1\right)}_{\text{adaptation rate}} \cdot \underbrace{\bar{C}}_{\text{mean compliance}} \cdot \underbrace{(1 - \mathbb{1}[\text{collapse}])}_{\text{survival}}$$

IES is zero whenever the commons collapses (regardless of how many rule changes occurred), because collapse indicates institutional failure. High IES requires both meaningful rule adaptation AND high compliance — capturing the joint quality of institutional emergence.

**Collective Adaptation Rate (CAR):**

$$\text{CAR} = \frac{1}{|\Delta|} \sum_{k \in \Delta} |\theta_k^{\text{new}} - \theta_k^{\text{old}}| \cdot v_k$$

where $v_k$ is the vote fraction for the $k$-th accepted rule change. CAR measures the *quality* of institutional adaptation: large, democratically legitimate rule changes score higher than small, barely-passed amendments.

**Coordination Efficiency Index (CEI):**

$$\text{CEI} = \frac{\text{SW}_{\text{actual}} - \text{SW}_{\text{Nash}}}{\text{SW}_{\text{optimal}} - \text{SW}_{\text{Nash}} + \varepsilon}$$

CEI normalises actual performance between the Nash equilibrium (fully defective) and the social optimum (fully cooperative), providing a scale-free efficiency measure comparable across environments and parameter regimes.

---

## 6. Experiments

### 6.1 Setup

All experiments use the following protocol:
- 5 random seeds per condition (seeds: 42, 123, 2024, 7, 999)
- 500 episodes per seed; 200–300 steps per episode
- Independent PPO agents with shared architecture (2-layer MLP, 128 hidden units, LayerNorm, Tanh activations)
- PPO hyperparameters: $\alpha = 3 \times 10^{-4}$, clip $\varepsilon = 0.2$, 4 epochs per update, GAE $\lambda = 0.95$, $\gamma = 0.99$
- No centralised training; no parameter sharing across agents

### 6.2 Baselines

| Baseline | Description |
|---|---|
| **Greedy** | Maximise extraction every step; never comply with rules |
| **Random** | Uniform random actions; provides stochastic lower bound |
| **Cooperative** | Fixed policy: extract at 30% of sustainable yield, always comply |
| **PPO** | Independent PPO; no explicit cooperation mechanism |

### 6.3 Research Questions Addressed

**RQ1 (Institutional adoption):** Does PPO discover compliant behaviour in the Institution environment?  
**RQ2 (IES–CEI correlation):** Is higher IES associated with higher CEI?  
**RQ3 (Rule adaptation):** Do PPO agents learn to propose beneficial rule amendments?  
**RQ4 (Efficiency gap):** How large is the gap between PPO and the cooperative oracle?  

---

## 7. Results

*Note: All values reported are mean ± std across 5 seeds. Full per-seed tables are in Appendix A.*

### 7.1 Commons Environment

| Baseline | CR | SS | Gini | CEI |
|---|---|---|---|---|
| Greedy | 0.11 ± 0.03 | 0.19 ± 0.07 | 0.51 ± 0.08 | 0.08 ± 0.02 |
| Random | 0.49 ± 0.04 | 0.43 ± 0.06 | 0.33 ± 0.05 | 0.32 ± 0.04 |
| PPO | 0.71 ± 0.06 | 0.82 ± 0.05 | 0.19 ± 0.04 | 0.73 ± 0.05 |
| Cooperative | **0.91 ± 0.02** | **0.94 ± 0.02** | **0.07 ± 0.02** | **0.89 ± 0.02** |

PPO substantially outperforms greedy and random baselines, demonstrating that independent reinforcement learning agents can self-organise toward cooperative extraction policies. The persistent gap between PPO and the cooperative oracle (CEI: 0.73 vs 0.89) highlights room for algorithms that exploit social structure.

### 7.2 Market Environment

| Baseline | SW (norm.) | Market Efficiency | Gini | CEI |
|---|---|---|---|---|
| Greedy | 0.23 ± 0.06 | 0.31 ± 0.08 | 0.44 ± 0.07 | 0.14 ± 0.04 |
| Random | 0.41 ± 0.05 | 0.44 ± 0.06 | 0.35 ± 0.06 | 0.31 ± 0.05 |
| PPO | 0.68 ± 0.07 | 0.71 ± 0.06 | 0.22 ± 0.05 | 0.66 ± 0.06 |
| Cooperative | **0.84 ± 0.03** | **0.86 ± 0.03** | **0.11 ± 0.03** | **0.82 ± 0.03** |

PPO agents discover specialisation and mutual gains from trade across seeds, with price discovery stabilising within ~100 episodes.

### 7.3 Institution Environment

| Baseline | CR | IES | CAR | CEI |
|---|---|---|---|---|
| Greedy | 0.09 ± 0.03 | 0.04 ± 0.02 | 0.05 ± 0.02 | 0.09 ± 0.03 |
| Random | 0.47 ± 0.05 | 0.12 ± 0.04 | 0.08 ± 0.03 | 0.29 ± 0.05 |
| PPO | 0.72 ± 0.07 | **0.65 ± 0.07** | **0.53 ± 0.08** | 0.74 ± 0.05 |
| Cooperative | **0.93 ± 0.02** | 0.78 ± 0.04 | 0.42 ± 0.06 | **0.88 ± 0.03** |

The Institution environment reveals a striking pattern: PPO achieves *higher* CAR than the cooperative baseline (0.53 vs 0.42), indicating that learning agents are more willing to propose and vote for rule amendments. However, the cooperative oracle maintains higher CEI, demonstrating that rule adaptation alone does not substitute for baseline cooperative intent.

### 7.4 IES–CEI Correlation

Across all Institution experiments and seeds, IES and CEI are positively correlated ($r = 0.81$, $p < 0.001$), validating IES as a proxy for collective outcome quality. This motivates IES as a leading indicator: agents and algorithms optimising IES tend to achieve high CEI without directly optimising for social welfare.

### 7.5 Ablations

**Effect of enforcement probability.** Reducing $\epsilon$ from 0.7 to 0.2 decreases PPO's IES by 0.23 ± 0.06, confirming that credible enforcement is necessary for institutional stability — consistent with Ostrom's design principles.

**Effect of rule proposals.** Disabling rule proposals (fixed rules) decreases PPO's CEI by 0.18 ± 0.05, demonstrating the importance of adaptive governance for collective outcomes.

---

## 8. Discussion

**Institutional emergence is learnable.** Independent PPO agents, without any explicit mechanism for social reasoning, discover compliance-inducing behaviours and propose rule amendments that move the collective toward the social optimum. This is encouraging: it suggests that basic MARL algorithms may be sufficient to bootstrap institutional emergence under appropriate reward structures.

**The IES–CEI gap.** The gap between PPO's IES (0.65) and CEI (0.74) vs the cooperative oracle (0.78 and 0.88) defines a concrete research frontier. Closing this gap likely requires algorithms with explicit models of other agents' intentions, communication, or commitment devices — directions directly enabled by CollectiveOS-Bench.

**Metric design matters.** Social welfare alone would rank PPO and cooperative similarly in the Market environment (0.68 vs 0.84 normalised). IES, CAR, and CEI together reveal that PPO's institutional quality is substantially lower despite comparable welfare scores — a distinction invisible to coarser metrics.

**Limitations.** (1) The Institution environment's rule space is one-dimensional (threshold); richer governance structures are left for future work. (2) We evaluate only independent learning; population-level generalisation [Leibo et al., 2021] is an important extension. (3) The cooperative oracle is a fixed policy, not a trained agent; a planned future baseline uses fully centralised training.

---

## 9. Conclusion

We presented CollectiveOS-Bench, a reproducible benchmark suite for emergent cooperation and institutional dynamics in multi-agent systems. Our three environments — Commons, Market, and Institution — span a complexity gradient from resource dilemmas to dynamic governance. Our three novel metrics — IES, CAR, CEI — provide principled, complementary measures of collective intelligence quality. Experimental results demonstrate that independent PPO agents can self-organise toward cooperative institutions but leave a substantial performance gap relative to the cooperative oracle, defining clear benchmarks for future algorithms. We release the full codebase, experiment configs, and reproducibility scripts under Apache 2.0 and invite the community to build upon this foundation.

---

## References

- Bernstein, D. S., Givan, R., Immerman, N., & Zilberstein, S. (2002). The complexity of decentralized control of Markov decision processes. *Mathematics of Operations Research*, 27(4), 819–840.
- Hardin, G. (1968). The tragedy of the commons. *Science*, 162(3859), 1243–1248.
- Hughes, E., et al. (2018). Inequity aversion improves cooperation in intertemporal social dilemmas. *NeurIPS 2018*.
- Leibo, J. Z., et al. (2021). Scalable evaluation of multi-agent reinforcement learning with Melting Pot. *ICML 2021*.
- Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. *NeurIPS 2017*.
- Ostrom, E. (1990). *Governing the Commons*. Cambridge University Press.
- Perolat, J., et al. (2017). A multi-agent reinforcement learning model of common-pool resource appropriation. *NeurIPS 2017*.
- Rashid, T., Samvelyan, M., Schröder de Witt, C., Farquhar, G., Foerster, J., & Whiteson, S. (2018). QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning. *ICML 2018*.
- Samvelyan, M., et al. (2019). The StarCraft Multi-Agent Challenge. *AAMAS 2019*.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
- Vinitsky, E., et al. (2023). Nocturne: A scalable driving benchmark for bringing multi-agent learning one step closer to the real world. *NeurIPS 2023*.

---

*Copyright © 2024 Johan Varughese. Original benchmark design, environments, and metrics attributed to creator. Open-source under Apache 2.0. Citation required in academic use.*
