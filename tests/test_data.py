import torch

from src.data.modular_addition import ModularAdditionDataset, train_test_split


def test_dataset_size():
    p = 113
    dataset = ModularAdditionDataset(p=p)
    assert len(dataset) == p * p, f"Expected {p*p}, got {len(dataset)}"


def test_all_labels_in_range():
    p = 113
    dataset = ModularAdditionDataset(p=p)
    labels = dataset.labels
    assert labels.min() >= 0, f"Min label {labels.min()} < 0"
    assert labels.max() < p, f"Max label {labels.max()} >= {p}"


def test_answer_correctness():
    p = 113
    dataset = ModularAdditionDataset(p=p)
    a, b = dataset.inputs[:, 0], dataset.inputs[:, 1]
    expected = (a + b) % p
    assert torch.equal(dataset.labels, expected), "Labels mismatch (a+b)%p"


def test_random_samples_correct():
    p = 113
    dataset = ModularAdditionDataset(p=p)
    rng = torch.Generator().manual_seed(0)
    indices = torch.randperm(p * p, generator=rng)[:100]
    for idx in indices:
        inp, label = dataset[idx]
        a, b = inp[0].item(), inp[1].item()
        expected = (a + b) % p
        assert label.item() == expected, \
            f"Sample ({a},{b}): expected {expected}, got {label.item()}"


def test_input_format():
    p = 113
    dataset = ModularAdditionDataset(p=p)
    inp, label = dataset[0]
    assert inp.shape == (3,), f"Input shape {inp.shape}, expected (3,)"
    assert inp[2].item() == p, f"Third token should be = (ID {p}), got {inp[2].item()}"
    assert inp[0] >= 0 and inp[0] < p
    assert inp[1] >= 0 and inp[1] < p


def test_train_val_no_leakage():
    p = 113
    seed = 42
    dataset = ModularAdditionDataset(p=p)
    train, val = train_test_split(dataset, val_split=0.3, seed=seed)

    train_ids = set()
    for i in range(len(train)):
        inp, _ = train[i]
        train_ids.add((inp[0].item(), inp[1].item()))

    val_ids = set()
    for i in range(len(val)):
        inp, _ = val[i]
        val_ids.add((inp[0].item(), inp[1].item()))

    overlap = train_ids & val_ids
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping (a,b) pairs between train and val"


def test_train_val_sizes():
    p = 113
    dataset = ModularAdditionDataset(p=p)
    train, val = train_test_split(dataset, val_split=0.3, seed=42)
    total = len(dataset)
    assert len(train) + len(val) == total, \
        f"Train ({len(train)}) + Val ({len(val)}) != Total ({total})"
    assert abs(len(val) / total - 0.3) < 0.001, \
        f"Val split {len(val)/total:.4f} != 0.3 (within 0.1% tolerance)"
