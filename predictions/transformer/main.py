import os
import sys

try:
    import stocknet
except ImportError:
    stocknet_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(stocknet_module_path)
    import stocknet

config_file = "./did_scaling.json"
stocknet.train_from_config(config_file)
