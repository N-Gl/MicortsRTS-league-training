# Implementation and evaluation of the reinforcement learning methods Self-Play and League Training in the MicroRTS environment

Presentation for the bachelor thesis

Niklas Glaser

Heinrich-Heine-Universität Düsseldorf

April 2026

---

# Overview

- Motivation
- Approach
- Experiments
- Results
- Discussion
- Conclusion

---

# Goals and Research Question

**Base Agent**

- combination of Behavioral Cloning (BC) and Proximal Policy Optimization (PPO)
- strong baseline, but weaknesses against several opponents

**Goal**

- improve the Base Agent with League Training (LT)
- training against a more diverse set of opponents
- focus on robustness and generalization

**Research Question**

- How much does LT with PPO improve robustness and generalization of the BC/PPO-pretrained agent?
- How strongly does this depend on the diversity of initial agents and BC experts?

---

# Setting

**MicroRTS**

- simplified RTS environment for RL research
- still contains resource management, unit production and combat
- supports controlled and repeatable experiments

**Main RL challenges**

- large combinatorial action space
- simultaneous decisions
- delayed and sparse rewards
- robustness against diverse opponents

Click the play area in Foxit Reader to start the embedded MicroRTS example video.

---

# Training Pipeline

BC → BC-FT → PPO → LT → PPO fine-tuning

- **BC / BC-FT:** imitate strong scripted opponents to obtain a competent starting policy
- **PPO:** refine the policy through exploration
- **LT:** train against a population of evolving opponents, not only bots or pure self-play
- **PPO fine-tuning:** repair the main remaining weakness after LT, especially against WorkerRushAI

---

# From SP to LT

Method extensions, not training phases.

- **SP:** train against the current version of the agent; cyclical strategies can appear, e.g. A beats B, B beats C, C beats A
- **FSP:** add past snapshots to reduce forgetting
- **PFSP:** add win-rate based sampling to focus on informative matchups
- **LT:** past policies stay relevant, exploiters target current weaknesses, and the objective shifts toward robustness across a population

**Takeaway**

LT changes who the agent trains against, not just how opponents are sampled.

---

# League Training Design

- **Main Agent:** primary policy optimized for final performance
- **Historical Agents:** frozen checkpoints kept in the opponent pool
- **Main Exploiters:** expose weaknesses close to the current main line
- **League Exploiters:** search for broader blind spots across the league

- **train continuously:** the Main Agent keeps improving throughout LT
- **freeze after checkpointing:** Historical Agents preserve older behaviours
- **reset and retrain:** exploiters repeatedly adapt to newly emerging weaknesses
- **vary opponent sets:** each role sees a different slice of the league

**Design principle**

Diversity in LT does not come from one self-play loop, but from combining train, freeze and reset behaviour across different roles.

---

# Bot Environments

- additional dynamic bot environments (CoacAI and Mayari)
- scripted bots: CoacAI, Mayari, WorkerRushAI, PassiveAI, LightRushAI, RandomAI and RandomBiasedAI
- MCTS based bots: Rojo, MixedBot, Izanagi, Tiamat, Droplet, GuidedRojoA3N and NaiveMCTS

---

# Observation Adjustment for SP

Player 2 observations are transformed so SP remains compatible with a Base Agent only trained as Player 1.

---

# LT Matchmaking

Matchmaking and opponent sampling in the LT setup.

---

# Other Implementation Changes

- delta score
- annealed entropy coefficient
- new initial agents

---

# Experiment Setup

- map: `basesWorkers16x16A`
- LT setup: 1 Main Agent, 4 Main Exploiters, 1 League Exploiter
- 25 environments per learning agent
- 5 to 10 additional dynamic bot environments
- approximately 130 million LT steps
- training time: about 100 hours
- maximum setup size: 160 parallel environments
- evaluations over at least 100 games against 12 opponents after LT and after fine-tuning

---

# Main Quantitative Result

- overall loss rate drops from **7.7%** to **2.2%**

---

# Evaluation Results

| Opponent | Main Agent without new initial agents | Base Agent | Main Agent (LT) | Main Agent + Fine-tuning |
|---|---:|---:|---:|---:|
| *bots used in training* |  |  |  |  |
| CoacAI | **100.0** | 96.0 | **100.0** | **100.0** |
| Mayari | **100.0** | 67.3 | **100.0** | **100.0** |
| *Built-in bots* |  |  |  |  |
| WorkerRushAI | 31.0 | **100.0** | 35.0 | 99.5 |
| PassiveAI | **100.0** | 99.7 | **100.0** | **100.0** |
| LightRushAI | **100.0** | 98.0 | **100.0** | **100.0** |
| RandomAI | **100.0** | 99.0 | **100.0** | **100.0** |
| RandomBiasedAI | 91.0 | 79.0 | 97.0 | **97.5** |
| *Other bots* |  |  |  |  |
| Rojo | **100.0** | 81.0 | **100.0** | **100.0** |
| MixedBot | 45.0 | 43.3 | 57.0 | **60.5** |
| Izanagi | 68.0 | 71.7 | 70.0 | **72.5** |
| Tiamat | 94.0 | 93.0 | **99.0** | **99.0** |
| Droplet | 45.0 | 67.7 | 62.0 | **75.5** |
| GuidedRojoA3N | 64.0 | 74.7 | 87.0 | **88.5** |
| NaiveMCTS | 0.0 | 0.7 | 2.0 | **6.5** |
| overall | 74.1 | 76.5 | 79.2 | **85.7** |

---

# Evaluation Results

| Opponent | Main Agent without new initial agents | Base Agent | Main Agent (LT) | Main Agent + Fine-tuning |
|---|---:|---:|---:|---:|
| *bots used in training* |  |  |  |  |
| CoacAI | **0.0** | 2.0 | **0.0** | **0.0** |
| Mayari | **0.0** | 27.3 | **0.0** | **0.0** |
| *Built-in bots* |  |  |  |  |
| WorkerRushAI | 63.0 | **0.0** | 55.0 | 0.5 |
| PassiveAI | **0.0** | **0.0** | **0.0** | **0.0** |
| LightRushAI | **0.0** | 2.0 | **0.0** | **0.0** |
| RandomAI | **0.0** | **0.0** | **0.0** | **0.0** |
| RandomBiasedAI | **0.0** | **0.0** | **0.0** | **0.0** |
| *Other bots* |  |  |  |  |
| Rojo | **0.0** | 5.3 | **0.0** | **0.0** |
| MixedBot | 33.0 | 34.3 | 20.0 | **13.5** |
| Izanagi | 15.0 | 16.3 | 7.0 | **6.0** |
| Tiamat | 6.0 | 7.0 | 1.0 | **0.5** |
| Droplet | 22.0 | **9.0** | 22.0 | 10.5 |
| GuidedRojoA3N | **0.0** | 1.7 | **0.0** | **0.0** |
| NaiveMCTS | **0.0** | 2.7 | **0.0** | **0.0** |
| overall | 9.9 | 7.7 | 7.5 | **2.2** |

---

# Interpretation of the Results

hypothesis:

- LT turns robustness against diverse opponents into part of the optimization target
- LT helps to find a better local optimum that is then further improved by fine-tuning, while pure PPO often gets stuck in not as generalized local optima.

---

# Player 1 Priority

**Observation**

MicroRTS gives player 1 action priority, so a potential concern is that self-play style training could accidentally favor player-1-specific behavior.

**Result from the thesis**

In one self-match evaluation of the same agent over 200 games, player 1 achieved 38% wins, 28% draws and player 2 achieved 34% wins.

- minimal effect onevaluations
- alternating the learning side to prevent the agent from exploiting player-1-specific priority

Open Player 1 Priority Example Video

---

# Main Limitation: Diversity

Base Agent vs Base Agent

Video example for the Base Agent's preferred strategy

Open Video

---

# Main Limitation: Diversity

- Exploiters initialized from similar policies tend to discover similar weaknesses.
- Some strategically different exploits are hard to reach because they require first unlearning the Base Agent's preferred behavior.
- New initial agents with focuses on different unit types to improve exploiter diversity and Main Agent generalization.
- Hypothesis: LT therefore depends strongly on the diversity already present at initialization

---

# Scope and Remaining Limitations

- evaluation is limited to one map: `basesWorkers16x16A`
- the setting uses full observability and no fog of war
- high computational cost
- exploiter diversity

**Takeaway**

The method works, but the strength of the final result depends strongly on the diversity and cost of the training setup.

---

# Future Work

- train on multiple maps and with fog of war
- improve the diversity and strength of initial agents
- reduce computational cost of MicroRTS training
- investigate stronger diversity mechanisms for exploiters

---

# Conclusion

**Conclusion**

LT with PPO improves the robustness and generalization of the MicroRTS Base Agent. Additional PPO fine-tuning improves the final agent even further.

**Key takeaway**

The effectiveness of LT depends strongly on exploiter diversity, which in turn depends on the diversity of the initial agents and BC experts.

- LT improves most matchups and the overall performance
- fine-tuning further improves the final agent
- diversity of initial agents is the critical practical factor

---

# Thank you

Questions?
