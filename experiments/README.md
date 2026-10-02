# experiments

Copies of the scripts that made each run's data, trained its adapter and evaluated it, as they were run;
the runs' outputs (data, adapters, evaluations, videos) are in `out/<run>/`, mostly links to
`/data001/maverick/jev-manip-out/`, and are not in git. Each script runs from the repo root.

| run | what |
|---|---|
| `vision/` | image-only reach and four-task (all_2cam) runs: evaluations and ablations |
| `ml10/`, `ml10_3cam/` | Meta-World ML10 split, two and three cameras, 20k and 40k rows per task |
| `mt50/` | the mt50 split (38 train tasks; T-comp, T-obj, T-vis), random looks; `PROMPTS.md` lists its task texts |
| `mt50co/` | co-training with language rows (lang_data.py); text and language-vs-action probes |
| `mt50instr/` | instruction-following executor (instr_data.py), from the mt50co adapter |
