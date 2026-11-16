# Plan (working notes)

IL first, RL on top, on Qwen3.5-4B. Evidence that this order works: Script2Play (arXiv 2611.11136, BC-only 4B beats
GPT-5.5 on unseen games), VPT (BC → RL reaches diamond tools RL alone can't), AlphaStar (supervised ~84th percentile
→ RL with KL to the supervised policy → Grandmaster), Tuyls et al. 2023 (IL return scales as a power law in compute),
Lumine (imitation + reasoning data plays Genshin's 5-h main story).

Eval ceiling: the scripted expert reaches 47/50 gold goals over 5 levels per game (mm2 misses 2, rivals 1).
Baselines on the gold level: gpt-6.1-sol 3/10, Qwen3.5-4B zero-shot 0/10, best old model 0/10
(results/old/summary.tsv).

Gates, so money isn't spent past a failure:
- after SFT: open-loop plan-switch > 0.6 and a 30-s gameplay check that it turns toward targets
- each DAgger round: kept only if its 5-seed eval beats the previous model
- RL: stop if win rates and curriculum k stay flat for 100 steps, or plan-switch drops below 0.5

## To do

Budget: ~$75-85 on Modal, hard-capped. Before anything: set the workspace spend limit in Modal (Settings → Usage),
log in (`.venv/bin/modal token new`), deploy, and run a ~$1 speed check of Qwen3.5-4B with the fla kernels to
replace the estimated step times below with measured ones.

```bash
cd ~/Desktop/baconhead/baconhead
BH_GPU=H100 BH_EVAL_GPU=A10 .venv/bin/modal deploy train/modal_app.py      # BH_GPU=A100-80GB on a Starter plan
```

- [ ] Data: ~20 h of whole games + ~10 h of skill clips, plans with directions (~$4 of CPU)

```bash
.venv/bin/modal run train/modal_app.py --cmd data --args "games --shards 48 --episodes 2 --ticks 1500" --name games_v1
.venv/bin/modal run train/modal_app.py --cmd data --args "skills --shards 48 --episodes 2 --clips 40" --name skills_v1
```

- [ ] Imitation, SFT: Qwen3.5-4B, ~6k steps, frozen vision (~$25-30). Gate: open-loop plan-switch > 0.6, and a
  30-s gameplay check that it turns toward targets

```bash
.venv/bin/modal run train/modal_app.py --cmd sft --args "--data /vol/data/games_v1,/vol/data/skills_v1 --out /vol/runs/il --max-steps 6000 --workers 14"
.venv/bin/modal run train/modal_app.py --cmd evaluate --args "--model /vol/runs/il --name il --openloop --play 1500 --seeds 5"
```

- [ ] Imitation, DAgger: 2 rounds on the model's own states (~$12-15); keep a round only if its eval beats the last

```bash
.venv/bin/modal run train/modal_app.py --cmd dagger --args "--model /vol/runs/il --base-data /vol/data/games_v1/shard_0000,/vol/data/skills_v1/shard_0000 --out /vol/runs/il_dagger --rounds 2 --per-game 12 --ticks 600 --sft-args '--max-steps 500'"
.venv/bin/modal run train/modal_app.py --cmd evaluate --args "--model /vol/runs/il_dagger/round1 --name il_dagger --openloop --play 1500 --seeds 5"
```

- [ ] RL on top: Backplay, end-only reward, KL to the IL model, ~300-400 steps (~$25-30). Stop if win rates and
  curriculum k stay flat for 100 steps, or plan-switch drops below 0.5 (RL rotted the plans last time)

```bash
.venv/bin/modal run train/modal_app.py --cmd rl --args "--model /vol/runs/il_dagger/round1 --out /vol/runs/rl --steps 400"
.venv/bin/modal run train/modal_app.py --cmd evaluate --args "--model /vol/runs/rl/last --name rl --openloop --play 1500 --seeds 5"
```

- [ ] Compare and watch: leaderboard, videos for every stage

```bash
.venv/bin/modal volume get --force baconhead /results/bench/ results/
.venv/bin/python -m train.eval --table
.venv/bin/python -m train.eval --video results/bench/play_rl
```

- [ ] Optional: GPT-written plans for the ~5-10k most decision-heavy states (gpt-6.1-sol sees the frame and the
  expert's action and writes the plan), if the IL model plans "what" but still not "where"

## Checks

```bash
.venv/bin/python -m sim.gold --check       # every gold goal still reachable by its hand-written run
.venv/bin/python -m model.vlm              # action/plan text codec round-trips
.venv/bin/python -m train.dataset skills --games garden --episodes 1 --clips 4 --out /tmp/skills   # data looks right
```
