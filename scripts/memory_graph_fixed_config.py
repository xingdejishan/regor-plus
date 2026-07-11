import copy
import json
from pathlib import Path

from easydict import EasyDict as edict


def load_fixed_config(path):
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as handle:
        protocol = json.load(handle)
    base_path = path.parent / protocol["base_config"]
    with open(base_path, encoding="utf-8") as handle:
        values = json.load(handle)
    values.update(copy.deepcopy(protocol["top_level_overrides"]))
    values["memory_graph"].update(copy.deepcopy(protocol["memory_overrides"]))
    return edict(values), protocol, path
