import os
import sys
import importlib.util

# ── Comprehensive NumPy BitGenerator Unpickling Compatibility Patch ───────────
try:
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
