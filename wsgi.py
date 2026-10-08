import os
import sys
import importlib.util

# ── Comprehensive NumPy Unpickling Compatibility Patch ────────────────────────
try:
    import sys
    import types
    import numpy as np

    if hasattr(np, "_core") and "numpy.core" not in sys.modules:
        _mc = types.ModuleType("numpy.core")
        _mc.numeric = getattr(np._core, "numeric", np._core)
        _mc.multiarray = getattr(np._core, "multiarray", np._core)
        _mc.umath = getattr(np._core, "umath", np._core)
        sys.modules["numpy.core"] = _mc
        sys.modules["numpy.core.numeric"] = _mc.numeric
        sys.modules["numpy.core.multiarray"] = _mc.multiarray
        sys.modules["numpy.core.umath"] = _mc.umath

    if hasattr(np, "core") and "numpy._core" not in sys.modules:
        _m = types.ModuleType("numpy._core")
        _m.numeric = getattr(np.core, "numeric", np.core)
        _m.multiarray = getattr(np.core, "multiarray", np.core)
        _m.umath = getattr(np.core, "umath", np.core)
        sys.modules["numpy._core"] = _m
        sys.modules["numpy._core.numeric"] = _m.numeric
        sys.modules["numpy._core.multiarray"] = _m.multiarray
        sys.modules["numpy._core.umath"] = _m.umath

    import numpy.random._pickle as _npr_pickle
    if hasattr(_npr_pickle, "BitGenerators") and isinstance(_npr_pickle.BitGenerators, dict):
        for k, v in list(_npr_pickle.BitGenerators.items()):
            _npr_pickle.BitGenerators[v] = v
            _npr_pickle.BitGenerators[str(v)] = v
            if hasattr(v, "__name__"):
                _npr_pickle.BitGenerators[v.__name__] = v
            if hasattr(v, "__module__") and hasattr(v, "__name__"):
                _npr_pickle.BitGenerators[f"{v.__module__}.{v.__name__}"] = v
except Exception:
    pass

current_dir = os.path.dirname(os.path.abspath(__file__))
website_dir = os.path.join(current_dir, 'mini_projects_website')
if website_dir not in sys.path:
    sys.path.insert(0, website_dir)

spec = importlib.util.spec_from_file_location('web_app', os.path.join(website_dir, 'app.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
app = mod.app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
