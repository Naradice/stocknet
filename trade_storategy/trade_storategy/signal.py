class Trend:
    key = "base"
    id = 0
    
    def __eq__(self, other):
        return other.id == self.id
    
class LongTrend(Trend):
    key = "long"
    id = 1
    
class ShortTrend(Trend):
    key = "short"
    id = -1
    
class Signal:
    key = "base"
    id = 0
    possibility = 0.0
    order_type = None
    is_buy = None
    is_close = False
    dev = None
    order_price = None
    option_info = None
    order_price = None
    
    def __init__(self, std_name) -> None:
        self.std_name = std_name
    
    def __eq__(self, other):
        attr = dir(other)
        #if('id' in attr and 'order_type' in attr):
        if(other and dir(self) == dir(other)):
            return other.id == self.id
        return False
    
    def __str__(self) -> str:
        return f"(key={self.key}, order_type={self.order_type}, possibility:{self.possibility}, is_buy:{self.is_buy}, order_price:{self.order_price}, dev:{self.dev})"
    
class BuySignal(Signal):
    key = "buy"
    id = 1
    order_type = "Market"
    is_buy = True
    
    def __init__(self, std_name, price:float=None, possibility:float=1.0) -> None:
        super().__init__(std_name)
        self.order_price = price
        self.possibility = possibility
    
class SellSignal(Signal):
    key = "sell"
    id = -1
    order_type = "Market"
    is_buy = False
    
    def __init__(self, std_name, price:float=None, possibility:float=1.0) -> None:
        super().__init__(std_name)
        self.order_price = price
        self.possibility = possibility
        
class BuyOrderSignal(Signal):
    key = "buy_order",
    id = 2
    order_type = "Order"
    is_buy = True
    
    def __init__(self, std_name, price:float, possibility:float=1.0) -> None:
        super().__init__(std_name)
        self.order_price = price
        self.possibility = possibility

class SellOrderSignal(Signal):
    key = "sell_order",
    id = -2
    order_type = "Order"
    is_buy = False
    
    def __init__(self, std_name, price:float, possibility:float=1.0) -> None:
        super().__init__(std_name)
        self.order_price = price
        self.possibility = possibility
        
class CloseSignal(Signal):
    "signal to close all position"
    key = "close"
    id = 10
    order_type = "Market"
    is_buy = None
    close = True
    
    def __init__(self, std_name, price:float=None, possibility:float=1.0) -> None:
        super().__init__(std_name)
        self.order_price = price
        self.possibility = possibility

class CloseBuySignal(Signal):
    "signal to close short position"
    key = "close_buy"
    id = 11
    order_type = "Market"
    is_buy = True
    close = True
    
    def __init__(self, std_name, price:float=None, possibility:float=1.0) -> None:
        super().__init__(std_name)
        self.order_price = price
        self.possibility = possibility
        
class CloseSellSignal(Signal):
    "signal to close short position"
    key = "close_sell"
    id = -11
    order_type = "Market"
    is_buy = False
    close = True
    
    def __init__(self, std_name, price:float=None, possibility:float=1.0) -> None:
        super().__init__(std_name)
        self.order_price = price
        self.possibility = possibility

class ValueRangeSignal(Signal):
    key = "value_range",
    id = 100
    order_type = None
    is_buy = None
    
    def __init__(self, std_name, pricees:list, possibilities:list) -> None:
        super().__init__(std_name)
        self.order_price = pricees#[High, Low]
        self.possibility = possibilities
    
    
class SignalInfo:
    
    def __init__(self, trend:Trend, is_buy:bool, order_type: str="Market", price:float=None, possibility:float=1.0):
        self.trend = trend
        if is_buy:
            if order_type == "Market":
                self.signal = BuySignal(possibility)
            elif order_type == "Order":
                self.signal = BuyOrderSignal(price, possibility)
        elif is_buy == False:
            if order_type == "Market":
                self.signal = SellSignal(possibility)
            elif order_type == "Order":
                self.signal = SellOrderSignal(price, possibility)
        elif type(price) == list:
                self.signal = ValueRangeSignal(pricees=price, possibilities=possibility)
        else:
            raise Exception(f"unkown signal info:: is_buy:{is_buy}, price: {price}")