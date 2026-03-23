import os, sys

_project_root = r'C:/Users/ksato/workspace/python/stock/stocknet'
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    import stocknet
except ImportError as _e:
    print(f'ImportError: {_e}', flush=True)
    sys.exit(1)

stocknet.train_from_config(r'C:/Users/ksato/workspace/python/stock/stocknet/model_configs/test.json')
