from asyncio.log import logger
import datetime
import threading
import time
from trade_storategy.storategies import Storategy
from logging import getLogger, config
import json
import os

class ParallelStorategyManager:
    
    def __init__(self, storategies:list, days=0, hours=0, minutes=0, logger = None) -> None:
        if logger == None:
            dir = os.path.dirname(__file__)
            try:
                with open(os.path.join(dir, './settings.json'), 'r') as f:
                    settings = json.load(f)
            except Exception as e:
                self.logger.error(f"fail to load settings file on storategy main: {e}")
                raise e
            logger_config = settings["log"]
            try:
                config.dictConfig(logger_config)
            except Exception as e:
                self.logger.error(f"fail to set configure file on storategy main: {e}")
                raise e
            self.logger = getLogger(__name__)
        else:
            self.logger = logger
            for storategy in storategies:
                storategy.logger = logger
            
        if type(storategies) == list and len(storategies) > 0:
            self.storategies = storategies
            self.__duration = datetime.timedelta(days=days, hours=hours, minutes=minutes)
            
    def __start_storategy(self, storategy: Storategy):
        interval_mins = storategy.interval_mins
        interval = interval_mins * 60
        
        base_time = datetime.datetime.now()
        target_min = datetime.timedelta(minutes=(interval_mins- base_time.minute % interval_mins))
        target_time = base_time + target_min
        sleep_time = datetime.datetime.timestamp(target_time) - datetime.datetime.timestamp(base_time) - base_time.second
        if sleep_time > 0:
            time.sleep(sleep_time)
        while datetime.datetime.now() < self.__end_time:
            base_time = time.time()
            signal = storategy.run()
            next_time = ((base_time - time.time()) % interval) or interval
            if signal and signal.order_type != None:
                print(interval_mins, signal, next_time)
                if signal.is_close:
                    results = []
                    if signal.is_buy == None:
                        results = storategy.client.close_all_positions()
                        self.logger.info(f"long positions are closed, remaining budget is {storategy.client.market.budget}")
                    elif signal.is_buy == True:
                        results = storategy.client.close_short_positions()
                        self.logger.info(f"long positions are closed, remaining budget is {storategy.client.market.budget}")
                    elif signal.is_buy == False:
                        results = storategy.client.close_long_positions()
                        self.logger.info(f"short positions are closed, remaining budget is {storategy.client.market.budget}")
                        
                    if len(results) > 0:
                        for result in results:
                            if result:
                                self.logger.info(f"closed result: {result}")
                if signal.is_buy:
                    position = storategy.client.open_trade(signal.is_buy, amount=1,price=signal.order_price, order_type=signal.order_type, symbol="USDJPY")
                    self.logger.info(f"long position is opened: {position}, remaining budget is {storategy.client.market.budget}")
                elif signal.is_buy == False:
                    position = storategy.client.open_trade(signal.is_buy, amount=1,price=signal.order_price, order_type=signal.order_type, symbol="USDJPY")
                    self.logger.info(f"short position is opened: {position}, remaining budget is {storategy.client.market.budget}")
            time.sleep(next_time)
            
    def start_storategies(self):
        self.__start_time = datetime.datetime.now()
        self.__end_time = self.__start_time + self.__duration
        
        for storategy in self.storategies:
            t = threading.Thread(target=self.__start_storategy, args=(storategy,), daemon=True)
            t.start()