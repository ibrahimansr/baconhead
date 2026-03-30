"""
DPO (Direct Preference Optimization) package for baconhead.

Trains GameSense ViT weights to encode gameplay preferences
(alive/playing > danger > dead) without a separate reward model.

Usage:
  python run_dpo.py --data episode_data --model game_sense.pt --epochs 3
"""
