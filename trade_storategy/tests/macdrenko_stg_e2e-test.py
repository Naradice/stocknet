import unittest, os, json, sys, datetime
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(module_path)
import trade_storategy as ts
from finance_client.client_base import Client
from finance_client.csv.client import CSVClient
from finance_client.utils.idcprocess import *

from logging import getLogger, config
try:
    with open(os.path.join(module_path, 'trade_storategy/settings.json'), 'r') as f:
        settings = json.load(f)
except Exception as e:
    print(f"fail to load settings file: {e}")
    raise e
logger_config = settings["log"]
log_file_base_name = logger_config["handlers"]["fileHandler"]["filename"]
log_path = f'./{log_file_base_name}_mrenko_test_{datetime.datetime.utcnow().strftime("%Y%m%d%H")}.logs'
logger_config["handlers"]["fileHandler"]["filename"] = log_path
config.dictConfig(logger_config)
logger = getLogger("trade_storategy.test")

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_source/bitcoin_5_2017T0710-2021T103022.csv'))

class Test(unittest.TestCase):
    def test_MACDWidthCSV(self):
        client = CSVClient(file=file_path, auto_index=True, start_index=7500, logger=logger)
        macd_p = MACDpreProcess()
        renko_p = RenkoProcess(window=120)
        st1 = ts.storategies.MACDRenko(client, interval_mins=0.1, renko_process=renko_p, macd_process=macd_p, logger=logger)
        manager = ts.ParallelStorategyManager([st1], minutes=10, logger=logger)
        manager.start_storategies()
        
if __name__ == '__main__':
    unittest.main()