---
layout: post
title:  "量化交易Day1: 初探ccxt和backtrader"
date:   2024-08-05
last_modified_at: 2024-08-05
categories: [量化交易]
---

## ccxt和backtrader是什么？

[ccxt](https://github.com/ccxt/ccxt)是一款针对于加密货币交易所的开源项目，它提供了一套统一的API接口，可以让用户在不同的交易所之间进行交易。我们这里使用ccxt提供的python接口进行交易。  
[backtrader](https://github.com/mementum/backtrader)是一个基于python的量化交易回测框架，简单易用。  

使用上述两个库，直接pip安装即可。  

```bash
pip install ccxt backtrader
```

## ccxt的使用

几个基本概念：  
1. exchange 交易所，包括binance，OKX等。此处我们使用binance的接口。  
2. symbol 交易对，如BTC/USDT，ETH/USDT等。前一个是基准货币，后一个是标价货币，例如BTC/USDT表示一个BTC的价格是多少USDT。

### 创建交易所(exchange)对象
  
首先，使用ccxt需要先创建一个交易所对象。以binance为例。   
binance获取数据需要使用代理。把proxy项替换为你自己的代理服务器和端口即可。  
enableRateLimit=True表示启用速率限制，即每次请求之间有一个最小的时间间隔。请求过于频繁可能会被封禁IP地址。

```python
exchange = ccxt.binance({
    "enableRateLimit": True,
    "proxy": {
        "http": "127.0.0.1:7890",
        "https": "127.0.0.1:7890"
    }
})
```

在ccxt中，只是获取市场数据，不需要使用用户的apikey（平台给每个账户提供的，需要用户自己在平台激活使用）。而如果想要进行交易或者账户查询等操作，则需要使用apikey。  

```python
exchange.apiKey = "your api key"
exchange.secret = "your secret key"

# test
balance = exchange.fetch_balance()
# 如果没有报错，说明apikey和secret正确
```

### 获取历史数据

使用ccxt查询历史数据，使用exchange.fetch_ohlcv函数。
ohlcv: open, high, low, close, volume.
此处我们演示获取日线数据。
此处由于每次从api获取的数据条数有限，需要我们多次获取。

```python
symbol = "BTC/USDT"
time_interval = '1d'
since_time = datetime(2021, 1, 1)
to_time = datetime(2024, 8, 1)

df = pd.DataFrame() 
while since_time < to_time:
    # 将since_time转换为unix时间戳
    since = exchange.parse8601(since_time.strftime("%Y-%m-%d %H:%M:%S"))
    # 获取数据, limit=500表示最多获取500条数据 
    data = exchange.fetch_ohlcv(symbol=symbol, 
                                timeframe=time_interval,
                                since=since,
                                limit=500)
    new_df = pd.DataFrame(data, dtype=float)
    new_df.rename(columns={0: 'MTS', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}, inplace=True)
    # 把时间戳转换为datetime格式
    new_df['candle_begin_time'] = pd.to_datetime(new_df['MTS'], unit='ms')
    # 时差
    new_df['candle_begin_time_GMT8'] = new_df['candle_begin_time'] + timedelta(hours=8)
    new_df = new_df[['candle_begin_time_GMT8', 'open', 'high', 'low', 'close', 'volume']]

    df = pd.concat([df, new_df], ignore_index=True)
    since_time = df['candle_begin_time_GMT8'].iloc[-1] + timedelta(days=1)

df.to_csv(dataset_path, index=False)
```

得到csv文件，共有六列分别为时间，开盘价，最高价，最低价，收盘价，成交量：  
```
candle_begin_time_GMT8,open,high,low,close,volume
2021-01-01 08:00:00,28923.63,29600.0,28624.57,29331.69,54182.925011
2021-01-02 08:00:00,29331.7,33300.0,28946.53,32178.33,129993.873362
2021-01-03 08:00:00,32176.45,34778.11,31962.99,33000.05,120957.56675
2021-01-04 08:00:00,33000.05,33600.0,28130.0,31988.71,140899.88569
2021-01-05 08:00:00,31989.75,34360.0,29900.0,33949.53,116049.997038
......
```

至此，我们已经学会了使用ccxt获取历史数据。ccxt可以获取数据的类型多种多样，这里我们仅仅用到ohlcv查询，更多请查阅文档。

## backtrader回测

backtrader的官方文档写的有些不太清楚（德国人写的英文文档可能是这样的），我跟随文档的quickstart做了一遍。下面是我自己的一些对官方文档中quickstart的解读。  

backtrader的运作围绕一个Cerebro（西班牙语的“大脑”）对象实例进行运作。大体流程如下：创建Cerebro实例。添加数据。添加策略。运行回测。  
其中，数据和策略都可以有多个。

### Cerebro的数据源：backtrader.feeds

backtrader的数据源是通过feeds来提供的，正如其名，我们相当于把数据源喂给Cerebro。feeds是一个抽象类，backtrader提供了读取yahoo财经、csv文件、pandas dataframe等数据源的feeds类。此处我们使用前面创建好的csv文件作为数据源，即使用backtrader.feeds.GenericCSVData来初始化数据源。  

值得一提的是，backtrader中数据的基本单位是“line”。这个line就是一个时间序列。例如我们的ohlcv数据共有5条lines：open, high, low, close, volume。

```python
data = btfeeds.GenericCSVData(
    dataname=dataset_path,
    timeframe=bt.TimeFrame.Minutes,
    fromdate=datetime(2021, 1, 1),
    todate=datetime(2024, 8, 1),
    nullvalue=0.0,
    dtformat=('%Y-%m-%d %H:%M:%S'),
    datetime=0,
    open=1,
    high=2,
    low=3,
    close=4,
    volume=5,
    openinterest=-1
)
```

参数的意思显而易见。注意，在backtrader中，除了ohlcv，默认还有一个line是持仓量（open interest），我们这里没有这个数据，所以设置为-1。

### Cerebro的策略：backtrader.Strategy

我们需要创建自己的策略了，直接继承bt.Strategy即可。

```python
class TestStrategy(bt.Strategy):
    params = (
        ("maperiod", 15), 
        ("printlog", False),
    )

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            logging.info("{}, {}".format(dt.isoformat(), txt))
            
    def __init__(self):
        # Keep a reference of the "close line" in the datas[0]
        self.dataclose = self.datas[0].close
        
        # Keep a track of orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        # Add a MovingAverageSimple indicator
        self.sma = btind.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)

        # Indicators for the plotting show
        btind.ExponentialMovingAverage(self.datas[0], period=25)
        btind.WeightedMovingAverage(self.datas[0], period=25, subplot=True)
        btind.StochasticSlow(self.datas[0])
        btind.MACDHisto(self.datas[0])
        rsi = btind.RSI(self.datas[0])
        btind.SmoothedMovingAverage(rsi, period=10)
        btind.ATR(self.datas[0], plot=False)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log("BUY EXECUTED, Price: {:.2f}, Cost: {:.2f}, Comm: {:.2f}".format(
                        order.executed.price,
                        order.executed.value,
                        order.executed.comm),
                    doprint=True
                )
                
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log("SELL EXECUTED, Price: {:.2f}, Cost: {:.2f}, Comm: {:.2f}".format(
                        order.executed.price,
                        order.executed.value,
                        order.executed.comm),
                    doprint=True
                )
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected", doprint=False)
            
        self.order = None 

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log("OPERATION PROFIT, GROSS: {:.2f}, NET: {:.2f}".format(
                trade.pnl, trade.pnlcomm),
            doprint=True
        )

    def next(self):
        self.log("Close, {:.2f}".format(self.dataclose[0]))
        
        if self.order:
            return
        
        if not self.position:
            if self.dataclose[0] > self.sma[0]:
                self.log("BUY CREATE, {:.2f}".format(self.dataclose[0]))
                self.order = self.buy()
                
        else:
            if self.dataclose[0] < self.sma[0]:
                self.log("SELL CREATE, {:.2f}".format(self.dataclose[0]))
                self.order = self.sell()
                
    def stop(self):
        self.log("MA Period: {}, Ending Value: {}".format(self.params.maperiod, self.broker.getvalue()), 
                 doprint=True)
```
让我们梳理一下这个策略的内容：  
这个策略是一个简单的均线策略，当价格上穿均线时，我们买入；当价格下穿均线时，我们卖出。  
1. params: 策略的参数。这是一个元组，定义了策略中需要用到的一些参数的默认值。例如此处，我们定义了一个maperiod参数，表示均线的周期。定义printlog为False，意思为默认情况下不需要打印日志。
2. log: 打印日志的函数。这个函数会在策略运行的过程中被调用，用于打印一些信息。例如我们在策略中调用了log函数，会打印出当前时间和价格。
3. __init__: 初始化函数。在这个函数中，我们定义了一些策略中需要用到的变量。例如self.dataclose表示数据源中的收盘价。self.order表示当前的订单。self.buyprice表示买入价格。self.buycomm表示买入手续费。self.sma表示均线指标。  
其中有一些需要注意的点：self.datas表示我们传入的所有策略的列表，而此处我们只传入了一组数据，那么我们想要使用这组数据就需要使用self.datas[0]。这里我们定义了self.dataclose为收盘价的简写，调用起来比较方便。我们这里也定义了一些指标（indicator），例如sma（移动平均线）。而后面的一些指标则是用于绘图。
4. notify_order: 订单通知函数。这个函数会在订单状态发生变化时被调用。例如订单被提交、订单被接受、订单被完成等。我们在这个函数中打印了订单的一些信息。订单状态有Submitted（提交）, Accepted（被broker接受）, Completed（完成）, Canceled, Margin（保证金不足）, Rejected（被拒绝）等。  
注：此处self.bar_executed变量的设置是为了知道订单在哪个bar被执行的，来计算持仓天数。len(self)就是策略执行的bar的数量，也就是已使用的数据的条数。
5. notify_trade: 交易通知函数。这个函数会在交易结束时被调用。我们在这个函数中打印了交易的一些信息。例如交易的盈利情况。**交易**的概念是指一次完整的买卖，一次交易可能需要由多个订单组成。
6. next: 下一个bar的函数。这个函数会在每个bar被调用。我们在这个函数中定义了策略的逻辑。例如我们在这个函数中判断了是否需要买入或者卖出。next是一个策略的核心方法。  
具体来说，策略执行的过程是一条一条的遍历数据，或者说数据一条一条地被喂进策略里。每喂进一条数据，我们就需要执行一次next函数。**在backtrader中，策略/数据以迭代器形式展开**，也就是说，self.dataclose[0]表示当前bar的收盘价，self.dataclose[-1]表示上一个bar的收盘价，self.dataclose[-2]表示上上个bar的收盘价，以此类推。
7. stop: 策略结束函数。这个函数会在策略结束时被调用。我们在这个函数中打印了策略的一些信息。例如策略的参数和最终的价值。实际上还有start函数，只不过我们此处省略了。
  
根据以上的步骤，我们基本上实现了一个非常简单的策略。

### Cerebro的其他杂项

```python
cerebro = bt.Cerebro()
# 添加策略
cerebro.addstrategy(TestStrategy)
# 添加数据
cerebro.adddata(data)
# 设置初始现金
cerebro.broker.setcash(1000)
# 设置佣金
cerebro.broker.setcommission(commission=0.001)
# 设置仓位管理器，这里我们使用固定的仓位，即每次交易0.01个比特币
cerebro.addsizer(bt.sizers.FixedSize, stake=0.01)
# 添加分析器，这里我们添加一个收益率分析器，后面会用到
cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
# 策略运行
cerebro.run()
# 策略绘图
cerebro.plot()
```

此处值得一提的是，cerebro和订单有关的操作都在cerebro.broker中，可以理解为cerebro中内置了一个证券经纪商，我们需要往经纪商账户中转入现金，设置其佣金等操作。

绘图结果：![回测结果](/assets/images/crypto/1-1.png)

除了内置的绘图工具，我们还需要一个结果分析器。我们直接使用了quantstats这个包进行分析（我们没有使用文档中推荐那个工具，那个有些太老了）。分析的结果我们可以选择以网页的形式展示出来。

```python
returns = cerebro.run()
pyfoliozer = result[0].analyzers.getbyname('pyfolio')
returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
qs.reports.html(returns, output='backtest.html', title='Backtest Report')
```

结果以网页的形式展示。

![backtest html](/assets/images/crypto/1-2.png)

## 完整代码

测试策略：backtest.py

```python
import pandas as pd
import numpy as np
import ccxt
import time
import os
from datetime import timedelta, datetime  
import backtrader as bt
import backtrader.indicators as btind
import backtrader.feeds as btfeeds
import argparse
from utils.logger import setup_logger 
import matplotlib.pyplot as plt
import logging
import quantstats as qs

parser = argparse.ArgumentParser()

parser.add_argument("--exchange", type=str, default="binance",
                    help="Name of the exchange")
parser.add_argument("--proxy", type=str, default="127.0.0.1:7890",
                    help="Proxy to use for the exchange")
parser.add_argument("--dataset-name", type=str, default="ohlcv-2024-08-05.csv",
                    help="the name of the dataset in the \"datesets\" folder")

def init_exchange(exchange_name, proxy, need_apikey):
    exchange_function = getattr(ccxt, exchange_name)
    exchange = exchange_function({"enableRateLimit": True, "proxies": proxy})
    if need_apikey:
        apikey_path = os.path.join("apikeys", exchange_name, "key.txt")
        with open(apikey_path, "r") as f:
            keys = f.readlines()
        exchange.apiKey = keys[0].strip()
        exchange.secret = keys[1].strip()
        try:
            balance = exchange.fetch_balance()
            total_balance = {key: value for key, value in balance['total'].items() if value != 0}
        except ccxt.AuthenticationError:
            logging.error("Authentication error. Please check your API keys.")
            exit(1)
        logging.info("Set up exchange with apikey, Account balance: {}".format(total_balance))
        return exchange
    logging.info("Set up exchange without apikey")  
    return exchange

class TestStrategy(bt.Strategy):
    params = (
        ("maperiod", 15), 
        ("printlog", False),
    )

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            logging.info("{}, {}".format(dt.isoformat(), txt))
            
    def __init__(self):
        # Keep a reference of the "close line" in the datas[0]
        self.dataclose = self.datas[0].close
        
        # Keep a track of orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        # Add a MovingAverageSimple indicator
        self.sma = btind.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)

        # Indicators for the plotting show
        btind.ExponentialMovingAverage(self.datas[0], period=25)
        btind.WeightedMovingAverage(self.datas[0], period=25, subplot=True)
        btind.StochasticSlow(self.datas[0])
        btind.MACDHisto(self.datas[0])
        rsi = btind.RSI(self.datas[0])
        btind.SmoothedMovingAverage(rsi, period=10)
        btind.ATR(self.datas[0], plot=False)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log("BUY EXECUTED, Price: {:.2f}, Cost: {:.2f}, Comm: {:.2f}".format(
                        order.executed.price,
                        order.executed.value,
                        order.executed.comm),
                    doprint=True
                )
                
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log("SELL EXECUTED, Price: {:.2f}, Cost: {:.2f}, Comm: {:.2f}".format(
                        order.executed.price,
                        order.executed.value,
                        order.executed.comm),
                    doprint=True
                )
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected", doprint=False)
            
        self.order = None 

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log("OPERATION PROFIT, GROSS: {:.2f}, NET: {:.2f}".format(
                trade.pnl, trade.pnlcomm),
            doprint=True
        )

    def next(self):
        self.log("Close, {:.2f}".format(self.dataclose[0]))
        
        if self.order:
            return
        
        if not self.position:
            if self.dataclose[0] > self.sma[0]:
                self.log("BUY CREATE, {:.2f}".format(self.dataclose[0]))
                self.order = self.buy()
                
        else:
            if self.dataclose[0] < self.sma[0]:
                self.log("SELL CREATE, {:.2f}".format(self.dataclose[0]))
                self.order = self.sell()
                
    def stop(self):
        self.log("MA Period: {}, Ending Value: {}".format(self.params.maperiod, self.broker.getvalue()), 
                 doprint=True)

if __name__ == "__main__":
    args = parser.parse_args()

    log_path = setup_logger(log_type="backtesting")
    logging.info("Using log path: {}".format(log_path))
    
    dataset_path = os.path.join("datasets", args.dataset_name)   
    logging.info("Using dataset path: {}".format(dataset_path))
    
    proxy = {"http": args.proxy, "https": args.proxy}
    logging.info("Using proxy:")
    logging.info(proxy)

    exchange_name = args.exchange
    exchange = init_exchange(exchange_name, proxy, 0)

    data = btfeeds.GenericCSVData(
        dataname=dataset_path,
        timeframe=bt.TimeFrame.Minutes,
        fromdate=datetime(2021, 1, 1),
        todate=datetime(2024, 8, 1),
        nullvalue=0.0,
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1
    )

    cerebro = bt.Cerebro()
    cerebro.addstrategy(TestStrategy)
    cerebro.adddata(data)
    cerebro.broker.setcash(1000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.FixedSize, stake=0.01)
    
    logging.info("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    returns = cerebro.run()

    logging.info("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
    cerebro.plot()
    plt.show()

    qs.reports.html(returns, output='backtest.html', title='Backtest Report')
```

get_dataset.py
  
```python
import pandas as pd
import numpy as np
import ccxt
import time
import os
from datetime import timedelta, datetime
import backtrader as bt
import backtrader.indicators as btind
import backtrader.feeds as btfeeds
import argparse
from utils.logger import setup_logger 
import logging

parser = argparse.ArgumentParser()

parser.add_argument("--exchange", type=str, default="binance",
                    help="Name of the exchange")
parser.add_argument("--proxy", type=str, default="127.0.0.1:7890",
                    help="Proxy to use for the exchange")
parser.add_argument("--trade", type=int, default=0,
                    help="If this script will trade (use the apikey to trade) (default 0)")
parser.add_argument("--dataset-name", type=str, default="ohlcv",
                    help="The name of the dataset")
parser.add_argument("--symbol", type=str, default="BTC/USDT",
                    help="The symbol of the market (default: BTC/USDT)")
parser.add_argument("--time-interval", type=str, default="15m",
                    help="The tiem interval of the data (default: 15m)")


def init_exchange(exchange_name, proxy, need_apikey):
    exchange_function = getattr(ccxt, exchange_name)
    exchange = exchange_function({"enableRateLimit": True, "proxies": proxy})
    if need_apikey:
        apikey_path = os.path.join("apikeys", exchange_name, "key.txt")
        with open(apikey_path, "r") as f:
            keys = f.readlines()
        exchange.apiKey = keys[0].strip()
        exchange.secret = keys[1].strip()
        try:
            balance = exchange.fetch_balance()
            total_balance = {key: value for key, value in balance['total'].items() if value != 0}
        except ccxt.AuthenticationError:
            logging.error("Authentication error. Please check your API keys.")
            exit(1)
        logging.info("Set up exchange with apikey, Account balance: {}".format(total_balance))
        return exchange
    logging.info("Set up exchange without apikey")  
    return exchange

if __name__ == "__main__":
    args = parser.parse_args()
    log_path = setup_logger(log_type="get_dataset")
    logging.info("Using log path: {}".format(log_path))

    dataset_name = "{}-{}.csv".format(args.dataset_name, datetime.now().strftime("%Y-%m-%d")) 
    dataset_path = os.path.join("datasets", dataset_name)
    
    proxy = {"http": args.proxy, "https": args.proxy}
    logging.info("Using proxy:")
    logging.info(proxy)
    exchange_name = args.exchange
    exchange = init_exchange(exchange_name, proxy, args.trade)

    symbol = args.symbol
    time_interval = args.time_interval
    
    time_interval = '1d'
    since_time = datetime(2021, 1, 1)
    to_time = datetime(2024, 8, 1)
    logging.info("Getting data from {} to {}".format(since_time, to_time))
    
    df = pd.DataFrame() 
    while since_time < to_time:
        since = exchange.parse8601(since_time.strftime("%Y-%m-%d %H:%M:%S"))
        data = exchange.fetch_ohlcv(symbol=symbol, 
                                    timeframe=time_interval,
                                    since=since,
                                    limit=500)
        new_df = pd.DataFrame(data, dtype=float)
        new_df.rename(columns={0: 'MTS', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}, inplace=True)
        new_df['candle_begin_time'] = pd.to_datetime(new_df['MTS'], unit='ms')
        new_df['candle_begin_time_GMT8'] = new_df['candle_begin_time'] + timedelta(hours=8)
        new_df = new_df[['candle_begin_time_GMT8', 'open', 'high', 'low', 'close', 'volume']]

        df = pd.concat([df, new_df], ignore_index=True)
        since_time = df['candle_begin_time_GMT8'].iloc[-1] + timedelta(days=1)
    
    df.to_csv(dataset_path, index=False)
    logging.info("Done.")
```

utils/logger.py 从[LibFewShot](https://github.com/RL-VIG/LibFewShot)抄过来的一个logger配置，觉得颜值很高。

```python
# -*- coding: utf-8 -*-
import logging
import os
from logging import config
from datetime import datetime   

try:
    USE_RICH_CONSOLE = True
    import rich
except ImportError:
    USE_RICH_CONSOLE = False

str_level_dict = {
    'NOTEST': logging.NOTSET,
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

def setup_logger(log_type: str = 'trade',
                logger_dir: str = 'storage',
                log_level: str = 'INFO',
                rank: int = 0
                ):
    if log_level not in str_level_dict:
        raise ValueError(f"Invalid log level: {log_level}")
    
    level = str_level_dict[log_level]
    file_name = "{}-{}.log".format(log_type, datetime.now().strftime("%b-%d-%Y-%H-%M-%S"))
    log_path = os.path.join(logger_dir, file_name)
    if rank == 0:
        logging_config = {
            "version": 1,
            "disable_existing_loggers": True,
            "formatters": {
                "simple": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "level": level,
                    "class": "logging.StreamHandler",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "level": level,
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "simple",
                    "filename": log_path,
                    "maxBytes": 100 * 1024 * 1024,
                    "backupCount": 3,
                },
            },
            "loggers": {
                "": {
                    "handlers": [
                        ("rich-console" if USE_RICH_CONSOLE else "console"),
                        "file",
                    ],
                    "level": level,
                    "propagate": True,
                }
            },
        }
    else:
        logging_config = {
            "version": 1,
            "disable_existing_loggers": True,
            "formatters": {
                "simple": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "level": level,
                    "class": "logging.StreamHandler",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "": {
                    "handlers": [
                        ("rich-console" if USE_RICH_CONSOLE else "console"),
                    ],
                    "level": level,
                    "propagate": True,
                }
            },
        }
        
    if USE_RICH_CONSOLE:
        logging_config["handlers"].update(
            {
                "rich-console": {
                    "level": level,
                    "class": "rich.logging.RichHandler",
                }
            }
        )

    logging.config.dictConfig(logging_config)
    return log_path
```

参考：
1. [20分钟讲明白最好用的开源量化回测框架BACKTRADER，流程，技巧，避坑](https://www.bilibili.com/video/BV1wN411a7Tz/?spm_id_from=333.337.search-card.all.click&vd_source=49de56206c51061afbede01d76ad6536) 
2. [官方文档](https://www.backtrader.com/docu/quickstart/quickstart/)