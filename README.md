# To do

- [ ] Raise the Modal spend limit on the `sadiasansari1` workspace, then log in and deploy

```bash
cd ~/Desktop/baconhead/baconhead
.venv/bin/modal token new
BH_GPU=H100 .venv/bin/modal deploy train/modal_app.py
```

- [ ] Finish the 3B SFT (resumes from checkpoint-1500 of 9000, ~3.6 h, ~$19)

```bash
.venv/bin/modal run train/modal_app.py --cmd sft --args "--data /vol/data/sim_sft_v2 --out /vol/runs/sft_v2 --max-steps 9000"
```

- [ ] Benchmark the 3B SFT with the open-loop check (~$1). Go on to DAgger only if its changed-keys accuracy is well above 0.54

```bash
.venv/bin/modal run train/modal_app.py --cmd eval --args "--model /vol/runs/sft_v2 --name qwen2.5-vl-3b_sft_v2 --openloop"
.venv/bin/modal volume get --force baconhead /results/bench/ results/
.venv/bin/python -m train.eval --table
```

- [ ] DAgger, 2 rounds (~$13); each round is benchmarked into `/vol/runs/dagger_v2/dagger.json`

```bash
.venv/bin/modal run train/modal_app.py --cmd dagger --args "--model /vol/runs/sft_v2 --base-data /vol/data/sim_sft_v2 --out /vol/runs/dagger_v2 --rounds 2 --sft-args '--max-steps 500'"
.venv/bin/modal volume get --force baconhead /runs/dagger_v2/dagger.json results/dagger_v2.json
.venv/bin/modal run train/modal_app.py --cmd eval --args "--model /vol/runs/dagger_v2/round1 --name qwen2.5-vl-3b_dagger2 --openloop"
```

- [ ] Promote the best model to the base the engine serves

```bash
.venv/bin/modal run train/modal_app.py --cmd promote --args "/vol/runs/dagger_v2/round1"
```

- [ ] Optional: GRPO on top, only if DAgger beats walking forward (stop if the eval doesn't improve or entropy collapses)

```bash
.venv/bin/modal run train/modal_app.py --cmd data --args "--shards 32 --episodes 2 --ticks 600 --states" --name sim_grpo
.venv/bin/modal run train/modal_app.py --cmd grpo --args "--model /vol/runs/dagger_v2/round1 --data /vol/data/sim_grpo --out /vol/runs/grpo_v4 --max-steps 300"
.venv/bin/modal run train/modal_app.py --cmd eval --args "--model /vol/runs/grpo_v4 --name qwen2.5-vl-3b_grpo300 --openloop"
```

- [ ] Put a fresh OpenRouter key in `.env` (`OPENROUTER_API_KEY=...`; the current one is expired), then benchmark API models

```bash
.venv/bin/python -m train.eval --api google/gemini-2.5-flash --name gemini-2.5-flash_zeroshot
.venv/bin/python -m train.eval --table
```

- [ ] Calibrate the real-Roblox drag timing (`LOOK_BUDGET_MS` in `control/controller.py`, `RDOWN_SETTLE_S` in `control/inject.py`) and try live play

```bash
.venv/bin/python run.py --backend modal
```

- [ ] Test the SSH backend

```bash
.venv/bin/python run.py --backend ssh --host user@gpu-box --dir ~/baconhead
```
