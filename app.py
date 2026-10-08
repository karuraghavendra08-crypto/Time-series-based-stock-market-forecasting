import os
import sys
import importlib.util

# ── Comprehensive NumPy Unpickling Compatibility Patch ────────────────────────
try:
    import sys
    import numpy as np

    if hasattr(np, "core"):
        if "numpy._core.numeric" not in sys.modules and hasattr(np.core, "numeric"):
            sys.modules["numpy._core.numeric"] = np.core.numeric
        if "numpy._core.multiarray" not in sys.modules and hasattr(np.core, "multiarray"):
            sys.modules["numpy._core.multiarray"] = np.core.multiarray
        if "numpy._core.umath" not in sys.modules and hasattr(np.core, "umath"):
            sys.modules["numpy._core.umath"] = np.core.umath

    if hasattr(np, "_core"):
        if "numpy.core.numeric" not in sys.modules and hasattr(np._core, "numeric"):
            sys.modules["numpy.core.numeric"] = np._core.numeric
        if "numpy.core.multiarray" not in sys.modules and hasattr(np._core, "multiarray"):
            sys.modules["numpy.core.multiarray"] = np._core.multiarray
        if "numpy.core.umath" not in sys.modules and hasattr(np._core, "umath"):
            sys.modules["numpy.core.umath"] = np._core.umath

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
