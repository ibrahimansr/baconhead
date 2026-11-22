import os
from copy import deepcopy

from baconhead import agent as A

EXEC = A.TICKS   # sim ticks per decision (300 ms)
KS = (2, 4, 8, 16, 32, 64)   # backplay: decisions before the expert's next milestone, easy to hard
_PROC = None


# ticks (multiples of EXEC) right before each score increase in the expert's episode
def milestones(game, seed, ticks=3000):
    from baconhead import sim
    env = sim.make(game, seed=seed, max_steps=10 ** 9)
    out, best = [], env.score()
    for t in range(ticks):
        env.step(env.expert())
        if env.score() > best + 1e-9:
            best = env.score()
            out.append((t // EXEC) * EXEC)
        if env.done:
            break
    return out


# the expert's episode replayed to `tick`: deterministic, so a row only stores (game, seed, tick)
def start_env(game, seed, tick):
    from baconhead import sim
    env = sim.make(game, seed=seed, max_steps=10 ** 9)
    for _ in range(tick):
        env.step(env.expert())
    return env


# reward 1 if the policy beats the milestone within k + slack decisions. act(env) -> (text, extra)
async def episode(game, seed, tick, k, act):
    env = start_env(game, seed, tick)
    score0, steps = env.score(), []
    for _ in range(k + max(4, k // 2)):
        text, extra = await act(env)
        steps.append(extra)
        A.play_chunk(env, *A.parse(text)[:4])
        if env.score() > score0 + 1e-9:
            return 1.0, steps
        if env.done:
            break
    return 0.0, steps


# miles custom generate: one episode -> one Sample per decision (previous + current frame), all sharing the episode's reward
async def generate(args, sample, sampling_params):
    global _PROC
    import torch
    from miles.rollout.sglang_rollout import generate as single_turn
    from baconhead.model.vlm import processor
    from baconhead.train.bc import frame
    _PROC = _PROC or processor(args.hf_checkpoint)
    md = sample.metadata
    prompt = _PROC.apply_chat_template(A.prompt(md["game"], [{"type": "image"}, {"type": "image"}]),
                                       add_generation_prompt=True, tokenize=False)
    prev = []

    async def act(env):
        img = frame(env)
        s = deepcopy(sample)
        s.prompt, s.multimodal_inputs = prompt, {"images": [(prev or [img])[-1], img]}
        prev.append(img)
        s.tokens, s.response, s.response_length, s.loss_mask, s.rollout_log_probs = [], "", 0, None, None
        s = await single_turn(args, s, sampling_params)
        mm = s.multimodal_train_inputs
        s.multimodal_train_inputs = {k: v for k, v in (mm or {}).items() if isinstance(v, torch.Tensor)} or None
        return s.response, s

    reward, samples = await episode(md["game"], md["seed"], md["tick"], md["k"], act)
    for s in samples:
        s.reward = reward
    return samples


# start states as (game, seed, tick, k), written easy to hard and read unshuffled
def prepare(out, games, seeds=8):
    import pandas as pd
    ms = {(g, s): milestones(g, s) for g in games for s in range(50_000, 50_000 + seeds)}
    rows = [{"prompt": g, "metadata": {"game": g, "seed": s, "tick": t - k * EXEC, "k": k}}
            for k in KS for (g, s), m in ms.items() for t in m if t >= k * EXEC]
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_parquet(f"{out}/train.parquet")
    print(f"{len(rows)} start states -> {out}/train.parquet")
    return f"{out}/train.parquet"


# fully-async GRPO on 2 H100s: GPU 0 trains (Megatron, CPU Adam), GPU 1 serves rollouts (SGLang)
def train(model, data, out, rollouts):
    from miles.utils.external_utils import command_utils
    args = (
        f"--hf-checkpoint {model} --load {model} --ref-load {model} --save {out}/ckpt --save-interval 50 "
        f"--save-hf {out}/hf/step_{{rollout_id}} "
        f"--prompt-data {data} --input-key prompt --metadata-key metadata --no-rollout-shuffle "
        "--custom-generate-function-path baconhead.train.rl.generate "
        f"--num-rollout {rollouts} --rollout-batch-size 16 --n-samples-per-prompt 8 --rollout-max-response-len 64 "
        "--rollout-temperature 0.8 --rollout-top-p 0.99 --use-dynamic-global-batch-size "   # episode length varies
        "--advantage-estimator grpo --eps-clip 0.2 --eps-clip-high 0.28 --use-tis "
        "--use-kl-loss --kl-loss-coef 0.01 --kl-loss-type low_var_kl --entropy-coef 0.0 "   # stay near the DAgger policy
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d "   # fp32 master + Adam = 54 GB, off the GPU
        "--train-backend megatron --megatron-to-hf-mode bridge --tensor-model-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 8192 "
        "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash "
        "--actor-num-nodes 1 --actor-num-gpus-per-node 1 --rollout-num-gpus 1 --num-gpus-per-node 2 "
        "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.8 "
        "--fully-async --pause-generation-mode in_place "
    )
    command_utils.default_config().create_backend().execute_train(
        train_args=args, num_gpus_per_node=2, megatron_model_type="qwen3.5-4B", train_script="train_async.py",
        extra_env_vars={"PYTHONPATH": "/root/Megatron-LM:/root"})


if __name__ == "__main__":   # reward sanity: from 2 decisions out, the expert's chunks beat the milestone and a no-op doesn't
    import asyncio
    from baconhead.agent import chunk_text
    from baconhead.train.bc import expert_chunk

    async def expert(env):
        return chunk_text(expert_chunk(env)), None

    async def noop(env):
        return "0 0 0 | | | | | |", None
    tick = milestones("obby", 50_000)[0] - 2 * EXEC
    won, _ = asyncio.run(episode("obby", 50_000, tick, 2, expert))
    lost, _ = asyncio.run(episode("obby", 50_000, tick, 2, noop))
    print(f"expert reward {won}, no-op reward {lost}")
    assert won == 1.0 and lost == 0.0
