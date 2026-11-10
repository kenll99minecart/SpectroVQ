"""Opt-in torch.compile for the SpectroVQ conv encoder and decoder.

Only ``model.encoder`` and ``model.decoder`` are compiled, in place through ``Module.compile()``:

* state_dict keys stay unchanged (no ``_orig_mod.`` prefix), so checkpoints load as before;
* the quantizer stays eager. Its control flow (k-means init, dead-code expiry, the random
  ``n_q`` dropout and ``use_vq`` switch) would otherwise cause graph breaks and one recompile per
  variant.

Compilation is lazy (on the first call). ``suppress_errors`` makes Dynamo fall back to eager if
it fails on a graph. ``dynamic=False`` keeps shapes static: the ragged last inference batch costs
one extra compile instead of switching to slower dynamic-shape kernels.
"""

import os
import sysconfig
import warnings

import torch

COMPILE_MODES = ("none", "default", "max-autotune")


def python_headers_available() -> bool:
    """Triton builds a C helper against Python.h: without it, every compile fails."""
    include = sysconfig.get_path("include")
    return os.path.exists(os.path.join(include, "Python.h"))


def compile_model(model, mode: str = "none", device=None, force: bool = False):
    """Compile ``model.encoder`` and ``model.decoder`` in place; return whether it did.

    Args:
        mode: one of COMPILE_MODES. "none" does nothing.
        device: the device the model runs on. Only CUDA is compiled unless ``force`` is set
            (Inductor on MPS is unreliable and the CPU gain is small).
        force: compile on any device (benchmarks, tests).
    """
    if mode not in COMPILE_MODES:
        raise ValueError(f"compile mode must be one of {COMPILE_MODES}, got {mode!r}")
    if mode == "none":
        return False
    device_type = torch.device(device).type if device is not None else "cpu"
    if device_type != "cuda" and not force:
        warnings.warn(
            f"--compile {mode} ignored on {device_type}: torch.compile is only used on CUDA.",
            stacklevel=2,
        )
        return False
    if not python_headers_available():
        # suppress_errors below would otherwise hide this: every graph falls back to eager.
        warnings.warn(
            f"--compile {mode} ignored: Python.h is missing, so torch.compile can't build its "
            "kernels. Install your Python's headers (RHEL: python3-devel).",
            stacklevel=2,
        )
        return False
    from torch import _dynamo

    _dynamo.config.suppress_errors = True  # fall back to eager instead of failing
    kwargs = {"dynamic": False}
    if mode != "default":
        kwargs["mode"] = mode
    model.encoder.compile(**kwargs)
    model.decoder.compile(**kwargs)
    return True
