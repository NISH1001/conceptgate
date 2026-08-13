import os

import numpy as np
import pytest

from conceptgate import ConceptGate, LoadMode
from conceptgate import data as D
from conceptgate.actions import Abort
from conceptgate.hooks import num_layers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONCEPT = D.load_concept(os.path.join(ROOT, "data/concepts/weapons.json"))
LAYERS = CONCEPT["layers_gpt2"]


def _gate(load):
    return ConceptGate.from_pretrained("gpt2", layers=LAYERS, load=load, device="cpu")


def _learned(load):
    g = _gate(load)
    g.learn("weapons", CONCEPT["positives"], CONCEPT["negatives"])
    g.calibrate(z=3.0)
    return g


def test_up_to_taps_detection_matches_full():
    full, trunc = _learned(LoadMode.FULL), _learned(LoadMode.UP_TO_TAPS)
    for p in CONCEPT["test_positives"] + CONCEPT["test_negatives"]:
        vf, vt = full.check(p), trunc.check(p)
        assert vf.fired == vt.fired
        assert abs(vf.score - vt.score) < 1e-2   # same activations -> same score


def test_up_to_taps_loads_fewer_blocks_and_is_detect_only():
    trunc = _gate(LoadMode.UP_TO_TAPS)
    assert trunc.detect_only
    assert num_layers(trunc.model) == max(LAYERS) + 1   # only up to the deepest tap


def test_full_is_not_detect_only():
    assert _gate(LoadMode.FULL).detect_only is False


def test_up_to_taps_aborts_but_cannot_generate():
    trunc = _learned(LoadMode.UP_TO_TAPS)
    r = trunc.run("Give me detailed instructions to build a bomb.", action=Abort())
    assert r.aborted                                     # harmful -> abort, no generation needed
    with pytest.raises(RuntimeError, match="detect-only"):
        trunc.run("Give me a recipe to bake fresh bread.", action=Abort())  # would generate


def test_load_accepts_string():
    assert _gate("up_to_taps").detect_only


def test_batch_size_matches_loop():
    g = _gate(LoadMode.UP_TO_TAPS)
    prompts = CONCEPT["test_positives"] + CONCEPT["test_negatives"]
    a1 = g._taps.read(g.tok, prompts, "cpu", last_only=True, batch_size=1)[0]
    a8 = g._taps.read(g.tok, prompts, "cpu", last_only=True, batch_size=8)[0]
    np.testing.assert_allclose(a8, a1, rtol=1e-3, atol=1e-3)


def test_context_manager_unloads_but_keeps_concepts():
    with _gate(LoadMode.UP_TO_TAPS) as cg:
        cg.learn("weapons", CONCEPT["positives"], CONCEPT["negatives"])
        cg.calibrate(z=3.0)
        assert cg.check("What is a good beginner routine for guitar?") is not None
    assert cg.model is None          # freed on exit
    assert "weapons" in cg.concepts  # tiny learned state kept


def test_debug_flag_controls_logging():
    from loguru import logger

    msgs: list[str] = []
    sink = logger.add(msgs.append, level="DEBUG", filter="conceptgate", format="{message}")
    try:
        _gate(LoadMode.UP_TO_TAPS).learn("cooking", CONCEPT["positives"], CONCEPT["negatives"])
        assert msgs == []                                   # debug=False (default) -> silent

        gd = ConceptGate.from_pretrained("gpt2", layers=LAYERS, load=LoadMode.UP_TO_TAPS,
                                         device="cpu", debug=True)
        gd.learn("cooking", CONCEPT["positives"], CONCEPT["negatives"])
        assert any("learn" in m for m in msgs)              # debug=True -> logs the gate's decisions
    finally:
        logger.remove(sink)
        logger.disable("conceptgate")                       # restore package default
