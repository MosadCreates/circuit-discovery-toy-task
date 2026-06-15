import sys
sys.path.insert(0, ".")

from src.utils.config import load_yaml, load_config, deep_merge


def test_load_default_config():
    cfg = load_yaml("configs/training/default.yaml")
    assert cfg["p"] == 113
    assert cfg["n_layers"] == 1
    assert cfg["n_heads"] == 4
    assert cfg["d_model"] == 128
    assert cfg["d_mlp"] == 512


def test_deep_merge_preserves_unset_keys():
    cfg = load_yaml("configs/training/default.yaml")
    overrides = {"p": 47, "n_heads": 2}
    merged = deep_merge(cfg, overrides)
    assert merged["p"] == 47
    assert merged["n_heads"] == 2
    assert merged["d_model"] == 128


def test_all_configs_load():
    for path in [
        "configs/training/default.yaml",
        "configs/training/small.yaml",
        "configs/training/fast-debug.yaml",
        "configs/patching/default.yaml",
        "configs/analysis/default.yaml",
    ]:
        cfg = load_yaml(path)
        assert "seed" in cfg or "p" in cfg  # at minimum


def test_small_config_has_smaller_prime():
    default = load_yaml("configs/training/default.yaml")
    small = load_yaml("configs/training/small.yaml")
    assert small["p"] < default["p"]
