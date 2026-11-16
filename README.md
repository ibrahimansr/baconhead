![baconhead](assets/banner.png)

# baconhead

A 4B vision-language model that plays Roblox games from pixels: imitation of scripted experts in ten simulated
games, then reinforcement learning on top.

<!-- VIDEO: the trained policy playing (replace with the final gameplay video / GIF) -->
**[ video of the trained policy playing — coming soon ]**

**Blog post:** _coming soon_ <!-- LINK: blog post -->

## Training

<!-- GRAPH: SFT / DAgger / RL loss and eval curves (replace with results image) -->
**[ loss and eval curves — coming soon ]**

## Run it

Weights: _coming soon on Hugging Face_ <!-- LINK: https://huggingface.co/<org>/<model> -->

```bash
pip install -r requirements.txt
python -m train.eval --model <hf-id-or-local-dir> --no-bench --play 600 --seeds 1   # the policy plays all 10 games
python -m train.eval --video results/bench/play_<name>                                # render its runs to mp4
```
