"""ConceptGate: a few-shot concept sidekick for frozen transformers.

Attach to any frozen model, learn concepts from ~10 examples, measure cheaply (a truncated
forward that runs only the tapped layers), and act via injected strategies:

    from conceptgate import ConceptGate
    from conceptgate.actions import Abort

    cg = ConceptGate.from_pretrained("gpt2", layers=[4, 6, 8])
    cg.learn("weapons", positives=[...], negatives=[...])
    cg.calibrate(z=3.0)
    cg.check(prompt)                    # Verdict — truncated forward
    cg.run(prompt, action=Abort())      # strategy decides; the gate drives + executes

Layers: `spectral`/`concept`/`mixture` are the numpy math core; `taps`/`hooks`/`data`
the torch boundary; `actions` the strategy layer; `gate` the facade. Only `ConceptGate` and
the actions are the surface you normally touch.
"""
from loguru import logger

# numpy-only, torch-free — safe to import for the math/toy path
from . import actions, concept, spectral  # noqa: F401
from .actions import Abort, ConceptAction, FireContext, Verdict  # noqa: F401
from .concept import BandpassConcept, Concept, ConceptBank, Direction  # noqa: F401

logger.disable("conceptgate")   # silent by default; ConceptGate(debug=True) re-enables

__all__ = [
    "ConceptGate", "RunResult", "LoadMode",           # lazy (torch) — see __getattr__
    "Verdict", "Abort", "ConceptAction", "FireContext",
    "Concept", "BandpassConcept", "ConceptBank", "Direction",
    "actions", "concept", "spectral",
]


def __getattr__(name):
    """Lazily expose the torch-backed facade so the numpy math path stays torch-free."""
    if name in ("ConceptGate", "RunResult", "LoadMode"):
        from . import gate

        return getattr(gate, name)
    raise AttributeError(f"module 'conceptgate' has no attribute {name!r}")
