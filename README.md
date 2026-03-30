<div align="center">

<img src="assets/banner.png" alt="baconhead" style="border-radius: 12px; width: 100%;" />

<br />
<br />

<a href="https://baconhead-page.vercel.app/">
  <img src="https://img.shields.io/badge/website-baconhead-black?style=for-the-badge&logo=vercel&logoColor=white" alt="Website" />
</a>
&nbsp;
<a href="https://discord.gg/AYyU5JZ4UK">
  <img src="https://img.shields.io/badge/discord-join-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord" />
</a>

<br />
<br />

**roblox bot that watches you play, learns game states from your gameplay, and takes over when you go idle.**

uses vision model we trained to understand what's on screen and plan actions. trains a local ViT model (GameSense) to detect death screens, danger zones, and menus — no hardcoded game knowledge, works on any roblox game.

</div>

---

## demo

<img src="assets/demo.gif" alt="baconhead demo" style="border-radius: 12px; width: 100%;" />

---

## how it works

1. you play roblox normally
2. after N seconds idle, the bot takes over
3. claude looks at your screen and decides what to do
4. GameSense (trained from your gameplay) detects deaths, danger, menus
5. you press any key and you're back in control

---

## setup

```bash
git clone https://github.com/your-username/baconhead.git
cd baconhead
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add your ANTHROPIC_API_KEY to .env
```

macos only rn. needs accessibility permission for keyboard/mouse control:
**System Settings → Privacy & Security → Accessibility** → add your terminal app.

---

## train GameSense (classification)

collect data by playing roblox (auto-labels frames from your gameplay):

```bash
python -m vision.collect --seconds 120 --out game_data
```

train the model:

```bash
python -m vision.train --data game_data --out game_sense.pt --epochs 10
```

prints per-class precision/recall when done. more data = better model.

---

## DPO fine-tuning (bake reward into weights)

after collecting gameplay data, run DPO to fold the reward signal
directly into the model weights — no separate reward model needed at inference time.

the preference signal comes from observed gameplay outcomes:
`playing > danger > menu > dead`
(chosen = frames from better states, rejected = frames from worse states).

```bash
# fine-tune existing game_sense.pt in-place
python run_dpo.py --data episode_data --model game_sense.pt --epochs 3

# save to a separate file instead
python run_dpo.py --data episode_data --model game_sense.pt \
    --out game_sense_dpo.pt --epochs 5 --beta 0.1

# smoke-test without saving
python run_dpo.py --data episode_data --model game_sense.pt --dry-run
```

the resulting checkpoint embeds a frozen **reference model** alongside the policy,
so future DPO rounds stay anchored to the original distribution (`--beta` controls
how far the policy can drift).

use the DPO-tuned model with the bot exactly as before:

```bash
python run_takeover.py --model game_sense.pt
```

---

## run

```bash
# basic — takes over after 3s idle
python run_takeover.py

# custom idle time + trained model
python run_takeover.py --idle 10 --model game_sense.pt

# monitor decisions to a log file
python run_takeover.py --idle 7 --model game_sense.pt --monitor bot_log.tsv

# no claude (random actions, for testing)
python run_takeover.py --no-scout

# capture only (no bot, just screen grab)
python run_capture.py --report --seconds 30
```

For the obby-beating implementation, use the `ObbyBeater` branch:

```bash
git checkout ObbyBeater
```

---

## contribute

1. fork this repo
2. create a branch (`git checkout -b my-feature`)
3. make your changes
4. open a PR

---

## todo

- [ ] collect more training data across different roblox games
- [ ] self-improving loop (bot labels its own gameplay data while running)
- [ ] add temporal context to GameSense (sequence of frames, not just one)
- [ ] obstacle avoidance from learned danger predictions
- [ ] multi-platform support (windows, linux — currently macos only)
- [ ] web dashboard for monitoring bot decisions live
- [ ] support custom action spaces beyond WASD
- [ ] fine-tune claude prompts based on per-game performance metrics
- [ ] replay buffer for offline RL training
