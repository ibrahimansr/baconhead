import numpy as np

from baconhead import agent as A
from baconhead.agent import chunk_text
from baconhead.train.bc import expert_chunk, frame, keep, row


# the model plays, the expert takes a chunk w.p. beta, every state is relabelled by the expert
# evaluating: held-out gold levels, beta 0, no rows, per-game goal / score / deaths instead
def rollout(llm, games, n_envs, decisions, beta, seed, evaluating=False):
    from vllm import SamplingParams
    from baconhead import sim
    from baconhead.sim.gold import PLANS, SEED
    from baconhead.train.dataset import _kw
    stop = [llm.get_tokenizer().convert_tokens_to_ids("<|im_end|>")]
    rng = np.random.default_rng(seed)
    envs = []
    for g in games:
        for k in range(n_envs):
            if evaluating:
                envs.append((g, sim.make(g, seed=SEED + k, max_steps=10 ** 9, **PLANS[g][0])))
            else:
                s = seed * 1000 + k
                envs.append((g, sim.make(g, seed=s, **_kw(g, np.random.default_rng(s)))))
    rows, texts, traces = [], [], []
    hit, prev = [None] * len(envs), None
    for d in range(decisions):
        labels = [expert_chunk(e) for _, e in envs]
        imgs = [frame(e, "cuda") for _, e in envs]
        prev = prev or imgs
        msgs = [A.prompt(g, [{"type": "image_pil", "image_pil": p}, {"type": "image_pil", "image_pil": im}])
                for (g, _), p, im in zip(envs, prev, imgs)]
        params = [SamplingParams(temperature=0.8, top_p=0.99, max_tokens=64, stop_token_ids=stop, skip_special_tokens=True,
                                 seed=seed * 100000 + d * 100 + i) for i in range(len(envs))]
        outs = llm.chat(msgs, params, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
        for i, ((g, e), p, im, o) in enumerate(zip(envs, prev, imgs, outs)):
            text = o.outputs[0].text
            mine, label = A.parse(text)[:4], labels[i]
            if not evaluating:
                rows.append(row(g, p, im, label))
                texts.append(chunk_text(label))
            if len(traces) < 400 and d % 10 == 0:
                traces.append({"game": g, "d": d, "model": text, "expert": chunk_text(label)})
            A.play_chunk(e, *(label if (not evaluating and rng.random() < beta) else mine))
            if evaluating and hit[i] is None and PLANS[g][2](e):
                hit[i] = d
        prev = imgs
    res = {"traces": traces}
    if not evaluating:
        return res, [r for r, k in zip(rows, keep(texts, rng)) if k]
    ob = [e for g, e in envs if g == "obby"]
    res |= {"obby_stage": float(np.mean([e.best_stage for e in ob])) if ob else None,
            "obby_deaths": float(np.mean([e.deaths for e in ob])) if ob else None,
            "obby_stages": [int(e.best_stage) for e in ob],
            "games": {g: {"goal": float(np.mean([h is not None for (gg, _), h in zip(envs, hit) if gg == g])),
                          "score": float(np.mean([e.score() for gg, e in envs if gg == g])),
                          "deaths": float(np.mean([e.deaths for gg, e in envs if gg == g]))} for g in games}}
    return res, []
