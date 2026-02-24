# ncap_priors

Sparse biological priors for NCAP swimmer control, grounded in Cook 2019 synapse counts and OpenWorm c302 3D geometry.

## What changed

### 1) Sparse prior generation (strict Cook->c302 translation, no dense matrices)
`generate_ncap_segment_priors(num_segments)` in `NMAP_amin/ncap_priors/swimmer_priors.py` now:
- loads two Cook 2019 sheets from:
  - `external/ConnectomeToolbox/cect/data/SI 5 Connectome adjacency matrices.xlsx`
  - chemical sheet: `"hermaphrodite chemical"`
  - gap sheet: `"herm gap jn symmetric"`
  - indices for both sheets:
    - `post_cells = df.iloc[2, 3:]`
    - `pre_cells = df.iloc[3:, 2]`
    - `syn_counts = df.iloc[3:, 3:]`
- parses real xyz coordinates from:
  - `external/ConnectomeToolbox/cect/data/c302_C2_FW.net.nml`
  - NeuroML `<location x= y= z=>`
- applies strict naming translation rules:
  - neurons: `DB01/VB01/DD01/VD01 -> DB1/VB1/DD1/VD1`
  - muscles: `dBWML1 -> MDL01`, `dBWMR1 -> MDR01`, `vBWML1 -> MVL01`, `vBWMR1 -> MVR01`
- computes 6 explicit sparse pathways:
  - `ipsi_db`: `DB01..DB07 -> dorsal muscles`
  - `ipsi_vb`: `VB01..VB11 -> ventral muscles`
  - `contra_db`: series conductance via `DD`
    - `N_eff = (N(DB->DD) * N(DD->ventral_muscle)) / (N(DB->DD) + N(DD->ventral_muscle))`
    - distance = direct `DB -> ventral muscle` Euclidean distance
  - `contra_vb`: series conductance via `VD`
    - `N_eff = (N(VB->VD) * N(VD->dorsal_muscle)) / (N(VB->VD) + N(VD->dorsal_muscle))`
    - distance = direct `VB -> dorsal muscle` Euclidean distance
  - `next_db`: `DB_i <-> DB_{i+1}` from gap sheet
  - `next_vb`: `VB_i <-> VB_{i+1}` from gap sheet
- returns scalar priors (not matrices/masks):
  - `dist_{ipsi_db, ipsi_vb, contra_db, contra_vb, next_db, next_vb}` normalized to `[0,1]`
  - `syn_{ipsi_db, ipsi_vb, contra_db, contra_vb, next_db, next_vb}`
  - counts and metadata

### 2) Model penalty path now sparse
`NMAP_amin/swimmer/models/tonic_ncap.py` now:
- removed dense segment-adapter / dense distance matrix dependence
- loads scalar sparse priors from `generate_ncap_segment_priors(...)`
- uses explicit sparse NCAP blocks (`use_weight_sharing=False`) and maps each pathway directly:
  - `ipsi_db -> muscle_d_d_*`
  - `ipsi_vb -> muscle_v_v_*`
  - `contra_db -> muscle_v_d_*`
  - `contra_vb -> muscle_d_v_*`
  - `next_db -> bneuron_d_prop_*`
  - `next_vb -> bneuron_v_prop_*`
- initializes pathway parameter blocks from pathway-specific synapse priors:
  - excitatory pathways initialized positive
  - `contra_*` pathways initialized negative (inhibitory)
- uses:
  - `L_prior = λ * Σ_pathway(dist_pathway * ||W_pathway||²)`

### 3) Agent penalty application now sparse
`NMAP_amin/swimmer/training/custom_tonic_agent.py` now:
- computes prior loss via `model.compute_topological_prior_loss(prior_reg_lambda)`
- no dense `NxN` topological matrix math in updater logic

### 4) Notebook fixed and aligned
`NMAP_amin/ncap_priors/demo_swimmer_priors.ipynb` now:
- fixes the `NameError` (`PROJECT_ROOT = candidate`, not `candidates`)
- uses the 6-pathway sparse prior API
- verifies notebook loss equals model loss using the same pathway mapping

### 5) Lean cleanup
- legacy dense-prior output path removed from active logic
- smoke tests updated for scalar sparse prior API:
  - `NMAP_amin/ncap_priors/tests/test_priors_smoke.py`
- package import surface now degrades gracefully when torch-backed wrappers are unavailable:
  - `NMAP_amin/ncap_priors/__init__.py` keeps sparse-prior utilities importable without forcing swimmer runtime deps

## Questions answered

1. **Why was the notebook failing with `NameError: candidates`?**  
Because of a typo in the root-discovery cell. Fixed to `PROJECT_ROOT = candidate`.

2. **Did we remove dense NxN regularization from active prior logic?**  
Yes. Sparse scalar pathway priors now drive regularization directly.

3. **Are penalties now based on real physical distances?**  
Yes. Distances are Euclidean xyz distances from `c302_C2_FW.net.nml`, normalized to `[0,1]`.

4. **Are next-neighbor B pathways now gap-junction based?**  
Yes. `next_db` and `next_vb` synapse priors now come from the Cook gap sheet (`herm gap jn symmetric`).

5. **Are contra pathways now series-conductance abstractions through D-neurons?**  
Yes. `contra_db` uses `DB->DD->ventral muscle` and `contra_vb` uses `VB->VD->dorsal muscle` with `N_eff` harmonic-combination formula.

## Validation performed

- `python3 -m py_compile` on modified Python files passed.
- manual smoke assertions equivalent to `NMAP_amin/ncap_priors/tests/test_priors_smoke.py` passed in local `nmap` environment.
- notebook-style manual sparse loss matches model sparse loss (`abs diff = 0.0`).

## Known constraints

- Translation rules are strict for the targeted neuron/muscle families and will need extension for other cell classes.
- If a pathway has no observed valid chain/edge in the loaded sheets, normalized distance defaults to `1.0` (tracked in metadata).

## Next steps

1. Add an integration test for one `CustomA2C` update step with nonzero `prior_reg_lambda` to ensure finite gradients under sparse priors.
2. Optionally expose `xlsx_path`, `chem_sheet`, `gap_sheet`, and `nml_path` arguments for controlled ablation runs.
3. If required for review scope, remove remaining legacy wrapper API in `NMAP_amin/ncap_priors/swimmer.py` to make package surface sparse-only.
