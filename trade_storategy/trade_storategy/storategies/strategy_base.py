import finance_client as fc
import trade_storategy as ts
from logging import getLogger, config
import json, os

class Storategy:

    key = "base"
    client: fc.Client = None
    
    def __init__(self, financre_client: fc.Client, interval_mins:int=-1, data_length:int = 100, logger=None) -> None:
        if logger == None:
            try:
                with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../settings.json')), 'r') as f:
                        settings = json.load(f)
            except Exception as e:
                self.logger.error(f"fail to load settings file on storategy: {e}")
                raise e
            
            logger_config = settings["log"]
            
            try:
                config.dictConfig(logger_config)
            except Exception as e:
                self.logger.error(f"fail to set configure file on storategy: {e}")
                raise e

            logger_name = "trade_storategy.storategy"
            self.logger = getLogger(logger_name)
        else:
            self.logger = logger
            
        self.client = financre_client
        self.data_length = data_length
        if interval_mins < 0:
            self.interval_mins = self.client.frame
        else:
            self.interval_mins = interval_mins
        self.trend = 0#0 don't have, 1 long_position, -1 short_position
             
        
    def run(self) -> ts.Signal:
        self.logger.debug("run base storategy for testing.")
        return ts.Signal(std_name="base")