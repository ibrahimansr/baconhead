import glob
import os

from baconhead import sim


# training-level settings: obby difficulty is drawn uniformly, so every difficulty is in the data
def _kw(game, rng):
    kw = {"max_steps": sim.ENVS[game].train_steps}
    if game == "obby":
        kw |= {"difficulty": float(rng.uniform(0, 1)), "respawn": 1.0}
    return kw


# comma-separated dataset dirs; a dir of shard_* subdirs loads as one
def load(spec):
    from datasets import concatenate_datasets, load_from_disk
    dirs = []
    for d in spec.split(","):
        dirs += [d] if os.path.exists(os.path.join(d, "dataset_info.json")) else sorted(glob.glob(os.path.join(d, "shard_*")))
    return concatenate_datasets([load_from_disk(d) for d in dirs])


def save(rows, out):
    import datasets
    if not rows:
        raise SystemExit("no rows")
    ds = datasets.Dataset.from_list(rows)
    ds.save_to_disk(out)
    print(f"{len(ds)} rows -> {out}")
    return ds
