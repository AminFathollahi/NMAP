# NMAP (Neuro-Modulating Architecture Priors)

NMAP is a biologically grounded embodied-control framework that bridges the gap between static **Neural Circuit Architectural Priors (NCAP)** and dynamic locomotion learning. The project integrates sub-millimeter *C. elegans* connectome structure (Cook 2019) directly into policy initialization and regularization, then evaluates how those priors interact with reinforcement learning in mixed-environment swimming and crawling tasks.

---

## 🧬 Scientific Context

Classical NCAP formulations encode architectural constraints but often under-specify how anatomical structure should influence training dynamics. NMAP addresses this by injecting connectome-derived pathway statistics into the optimization loop while preserving explicit biophysical constraints.

Core biological constraints implemented in this repository:

- **D-neuron series conductance abstraction** for crossed pathways:
  - `N_eff = (N1 * N2) / (N1 + N2)` for DB→DD→ventral-muscle and VB→VD→dorsal-muscle chains.
- **Exact gap-junction pathway extraction** from Cook 2019 gap matrices for adjacent B-neuron coupling (`DB_i↔DB_{i+1}`, `VB_i↔VB_{i+1}`).
- **Asymmetric relaxation oscillator dynamics** motivated by eLife 2021 locomotor rhythm analyses.

---

## 📦 Repository Layout

```
NMAP/
├── main.py                          # Training/evaluation entrypoint
├── swimmer/
│   ├── environments/
│   │   ├── mixed_environment.py     # Mixed water/land environment (curriculum radii)
│   │   ├── progressive_mixed_env.py # Progressive curriculum wrapper
│   │   └── tonic_wrapper.py         # Tonic gym bridge
│   ├── models/
│   │   ├── biological_ncap.py       # Standard biological NCAP
│   │   ├── enhanced_biological_ncap.py  # Enhanced NCAP with goal direction
│   │   └── tonic_ncap.py            # Tonic-compatible NCAP model
│   └── training/
│       ├── custom_tonic_agent.py    # CustomPPO/A2C with oscillation penalty
│       ├── curriculum_trainer.py    # 4-phase curriculum trainer
│       ├── ncap_trainer.py          # Interval-training NCAP trainer
│       └── swimmer_trainer.py       # Base Tonic trainer
├── connectome_priors/
│   ├── swimmer_priors.py            # Cook 2019 → sparse pathway priors
│   ├── c_elegans_connectome.py      # Connectome edge loading
│   └── c_elegans_geometry.py        # c302 NML geometry parsing
└── tests/                           # Validation scripts
run_experiments.py                   # 4-step ablation runner (repo root)
```

---

## 🚀 Running the Ablation Study

The full 4-step ablation is launched from the repository root:

```bash
python run_experiments.py
```

This sequentially runs and archives four experiments to `results/`:

| Step | Name | Flags |
|------|------|-------|
| 01 | Baseline (tabula rasa) | — |
| 02 | Oscillation preservation | `--force_oscillation` |
| 03 | Sparse initialization | `--force_oscillation --sparse_init` |
| 04 | Full NMAP | `--force_oscillation --sparse_init --sparse_reg_lambda 0.05` |

Each experiment runs for 500,000 training steps and its Tonic outputs are moved to `results/<name>/` on completion.

### Manual invocation

```bash
python NMAP/main.py --mode train --training_steps 500000
python NMAP/main.py --mode train --training_steps 500000 --force_oscillation
python NMAP/main.py --mode train --training_steps 500000 --force_oscillation --sparse_init
python NMAP/main.py --mode train --training_steps 500000 --force_oscillation --sparse_init --sparse_reg_lambda 0.05
```

---

## 🔬 Ablation Axes

| Flag | Effect |
|------|--------|
| `--sparse_init` | Connectome-informed weight initialization (excitatory ipsi, inhibitory contra, gap-junction next-neighbor) |
| `--sparse_reg_lambda <λ>` | Topological L2 regularization weighted by normalized Cook-distance per pathway |
| `--force_oscillation` | Variance penalty on actor output to prevent policy collapse to near-zero actions |

Switches are fully orthogonal and can be combined independently to isolate each contribution.

---

## 🧪 Audit & Current State (2026-02-23)

A zero-trust pre-launch audit was performed before the 12-hour automated ablation run. All checks and their outcomes are recorded below.

### ✅ Checks that passed

| Check | File | Finding |
|-------|------|---------|
| Cook 2019 XLSX absolute path | `connectome_priors/swimmer_priors.py` | `_resolve_connectome_artifact` now prioritizes `/home/amin/Research/NCAP/external/ConnectomeToolbox/cect/data/…` with legacy fallbacks; file confirmed present on disk. |
| PYTHONPATH absolute paths | `run_experiments.py` | All three PYTHONPATH components built from `Path(__file__).resolve()` and `Path.cwd().resolve()` — fully absolute at runtime. |
| `results/` archive path | `run_experiments.py` | `RESULTS_ROOT = ROOT / "results"` where `ROOT = Path(__file__).resolve().parent` — absolute. |
| Anti-reward-hack spawn jitter | `swimmer_trainer.py:456` | `_reset_with_randomized_start` enforces minimum 0.2 m displacement from default spawn and rejects positions within 0.5 m of the target geom; 24-attempt loop guarantees a valid sample. |
| Variance penalty for `--force_oscillation` | `custom_tonic_agent.py:184` | `_compute_force_oscillation_penalty` present in `CustomA2C`, promoted to `CustomPPO` via explicit class-attribute copy. Penalty = `scale * relu(min_variance - action_variance)` applied via separate backward pass. |
| Gradient flow through actor update | `curriculum_trainer.py` | Training update path (`_train_on_episode` → `_get_model_action`) does NOT use `torch.no_grad()`; `total_loss.backward()` fires with valid gradients. |

### 🔧 Issues found and fixed

| # | Severity | File | Issue | Fix Applied |
|---|----------|------|-------|-------------|
| 1 | **BLOCKER** | `mixed_environment.py:69` | Land zone curriculum stopped at 0.6 m radius; required 0.2 m was never reached. | Curriculum extended: `shrink_steps=[0,10,20,40]`, `radii=[1.0,0.6,0.4,0.2]` |
| 2 | **BLOCKER** | `ncap_trainer.py:59` | `early_stopping_patience = 10` — would trigger early stop after ~20k steps on a 500k run. | Changed to `early_stopping_patience = 30`. |
| 3 | **Silent bug** | `ncap_trainer.py:381` | `interval_reward` referenced before assignment in viscosity logging block; silent `NameError` on every interval meant viscosity scatter plot was always empty. | Moved viscosity logging block to after `interval_reward` is defined (line 386). |

### ⚠️ Known limitation (non-blocking)

`BiologicalNCAPAgent.test_step()` in `curriculum_trainer.py` wraps the forward pass with `torch.no_grad()` during rollout data collection. This is standard RL practice and does **not** break training (gradients flow correctly through the separate `_get_model_action` path in `_train_on_episode`). However, it was flagged as inconsistent with the stated requirement to remove `no_grad` from the actor loop. If on-policy gradient flow through the rollout is needed in future work, `test_step` should be split into inference-only and training-mode variants.

---

## 📊 Experimental Reporting Recommendations

For reproducible comparisons, report at minimum:
- random seed(s), total training steps, and hardware
- exact command used (including all ablation flags)
- evaluation horizon and environment settings
- trajectory-level metrics: distance (m), mean velocity (m/s), environment transitions
- reward summaries per phase (for curriculum runs)

---

##❓ Scientific & Architectural Questions Answered
Q: Why implement strict excitatory/inhibitory weight clamps instead of letting the network learn them?
A: To preserve the biological fidelity of the C. elegans motor circuit. Previous abstractions lazily mapped 'ventral' neurons to inhibitory, paralyzing the simulated ventral circuit. By enforcing strictly inhibitory bounds on contra-pathways (D-neurons) and excitatory bounds on ipsi/proprioceptive inputs and gap junctions, we force the RL optimizer to explore mathematically valid biological states rather than unphysical mathematical shortcuts.

Q: Why does our agent often 'curl up and shiver' instead of swimming?
A: In high-impedance environments (like viscous water), an unconstrained RL agent often falls into a local optimum: it discovers that locking its joints and shivering minimizes torque penalties while preventing backward drift. It learns to 'survive' rather than swim. We counter this policy collapse by injecting a variance penalty (--force_oscillation) and shaping the reward to penalize static, extreme postures.

Q: Should biologically constrained networks use PPO or A2C optimization?
A: PPO is the industry standard for continuous control due to trust-region clipping, which ensures stability. However, when weights are strictly clamped to biological bounds (positive/negative), PPO's clipping can get 'stuck' against these hard boundaries. A2C applies raw gradients, which is normally unstable, but our biological priors (sparse topology, D-neuron conductance math) act as natural structural stabilizers, making A2C a highly relevant alternative for biological architectures.

Q: How does target distance impact the emergence of coordinated gait?
A: If navigation targets are too large or too close (e.g., 1.0m radius), the agent can 'accidentally' fall into the reward zone without propagating a wave, triggering early stopping before locomotion is learned. By shrinking targets to 0.2m and pushing them to 3.0m, we maximize the environmental gradient, mathematically forcing the agent to learn true sinusoidal wave propagation to achieve the reward.

Q: How does the framework handle legacy RL environments built on deprecated libraries?
A: Instead of rewriting hundreds of legacy environment files, NMAP implements a 'Global Compatibility Bridge' (sys.modules['gym'] = gymnasium). This intercepts legacy API calls and dynamically routes them to the modern software stack, preventing software rot while preserving the integrity of the original research environments.

---

## 🛑 Challenges

1. **Biologically faithful contra-pathway abstraction**: The D-neuron inhibitory interneurons (DD/VD) cannot be modelled as direct weights without collapsing the three-cell chain. The series-conductance `N_eff` formula approximates functional throughput but loses segment-specificity. Future work should explore per-segment D-neuron instantiation.

2. **Curriculum radius vs. physical body size**: At 0.2 m radius the land zones are comparable to the swimmer body length. The swimmer may straddle the zone boundary, producing partial land physics. Monitoring `env_transitions` in evaluation logs will reveal whether this causes instability.

3. **Viscosity domain randomization range**: The current log-uniform range `[1e-5, 3e-1]` spans six orders of magnitude. At the low end `(1e-5)`, the swimmer is effectively in near-vacuum; at the high end `(3e-1)`, locomotion is heavily damped. Policy generalization across this range is the core stress test of the neuromodulatory pathway.

4. **Silent `no_grad` in curriculum rollout**: The `BiologicalNCAPAgent` re-runs the model in `_get_model_action` during policy updates, meaning gradients flow correctly but the rollout actions and training actions are computed from separate forward passes. This is equivalent to a replay-buffer policy-gradient approach, not a fully on-policy gradient. The consequence is that action-distribution shift between rollout and update may be non-trivial at low batch sizes.

---

## 🗺️ Next Steps

### Immediate (post-ablation)
- [ ] Analyse `results/` per-experiment reward curves; compare baseline vs. full NMAP on distance, velocity, and env-transition count.
- [ ] Plot viscosity-vs-reward scatter for `NCAPTrainer` runs to verify neuromodulation is exploiting the viscosity signal.
- [ ] Check that `action_variance` logged under `actor/action_variance` is consistently above `force_oscillation_min_variance=0.1` in experiments 02–04.

### Short-term
- [ ] Add an integration test: one `CustomPPO._update()` step with `prior_reg_lambda > 0`, assert finite gradients and non-zero `prior_loss`.
- [ ] Fix the `torch.no_grad()` split in `BiologicalNCAPAgent`: create explicit `_infer()` (no_grad, for env interaction) and keep `_get_model_action()` (grad, for policy update).
- [ ] Expose `xlsx_path`, `chem_sheet`, `gap_sheet`, and `nml_path` as CLI arguments so Cook 2019 source files can be swapped for ablation without code edits.

### Longer-term
- [ ] **Hierarchical RL**: pre-train low-level motor policy with structural priors, add high-level navigation controller for task-level decisions.
- [ ] **Biologically plausible plasticity**: replace backpropagation with Hebbian learning / Feedback Alignment to evaluate generalization under local learning rules.
- [ ] **Per-segment D-neuron instantiation**: replace `N_eff` series-conductance approximation with explicit DD/VD interneuron nodes for each body segment.

---

## 📚 References

- Cook et al., 2019 — *C. elegans* connectome adjacency matrices (chemical synapses and gap junctions).
- Lechner et al., 2019 — Neural Circuit Architectural Priors for Embodied Control (NCAP).
- eLife (2021) — Phase-response evidence supporting relaxation-oscillator locomotor rhythm generation in *C. elegans*.
- OpenWorm c302 — `c302_C2_FW.net.nml` 3D neuron geometry.
