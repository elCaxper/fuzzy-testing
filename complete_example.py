import pandas as pd
import numpy as np

# https://gist.github.com/ultragtx/6831eb04dfe9e6ff50d0f334bdcb847d
# calculating RSI (gives the same values as TradingView)
# https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas
def RSI(series, period=14):
    delta = series.diff().dropna()
    ups = delta * 0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[period - 1]] = np.mean(ups[:period])  # first value is sum of avg gains
    ups = ups.drop(ups.index[:(period - 1)])
    downs[downs.index[period - 1]] = np.mean(downs[:period])  # first value is sum of avg losses
    downs = downs.drop(downs.index[:(period - 1)])
    rs = ups.ewm(com=period - 1, min_periods=0, adjust=False, ignore_na=False).mean() / \
         downs.ewm(com=period - 1, min_periods=0, adjust=False, ignore_na=False).mean()
    return 100 - 100 / (1 + rs)


# calculating Stoch RSI (gives the same values as TradingView)
# https://www.tradingview.com/wiki/Stochastic_RSI_(STOCH_RSI)
def StochRSI(series, period=14, smoothK=3, smoothD=3):
    # Calculate RSI
    delta = series.diff().dropna()
    ups = delta * 0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[period - 1]] = np.mean(ups[:period])  # first value is sum of avg gains
    ups = ups.drop(ups.index[:(period - 1)])
    downs[downs.index[period - 1]] = np.mean(downs[:period])  # first value is sum of avg losses
    downs = downs.drop(downs.index[:(period - 1)])
    rs = ups.ewm(com=period - 1, min_periods=0, adjust=False, ignore_na=False).mean() / \
         downs.ewm(com=period - 1, min_periods=0, adjust=False, ignore_na=False).mean()
    rsi = 100 - 100 / (1 + rs)

    # Calculate StochRSI
    stochrsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    stochrsi_K = stochrsi.rolling(smoothK).mean()
    stochrsi_D = stochrsi_K.rolling(smoothD).mean()

    return stochrsi, stochrsi_K, stochrsi_D


# calculating Stoch RSI
#  -- Same as the above function but uses EMA, not SMA
def StochRSI_EMA(series, period=14, smoothK=3, smoothD=3):
    # Calculate RSI
    delta = series.diff().dropna()
    ups = delta * 0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[period - 1]] = np.mean(ups[:period])  # first value is sum of avg gains
    ups = ups.drop(ups.index[:(period - 1)])
    downs[downs.index[period - 1]] = np.mean(downs[:period])  # first value is sum of avg losses
    downs = downs.drop(downs.index[:(period - 1)])
    rs = ups.ewm(com=period - 1, min_periods=0, adjust=False, ignore_na=False).mean() / \
         downs.ewm(com=period - 1, min_periods=0, adjust=False, ignore_na=False).mean()
    rsi = 100 - 100 / (1 + rs)

    # Calculate StochRSI
    stochrsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    stochrsi_K = stochrsi.ewm(span=smoothK).mean()
    stochrsi_D = stochrsi_K.ewm(span=smoothD).mean()

    return stochrsi, stochrsi_K, stochrsi_D

# Ichimoku code: https://stackoverflow.com/a/61282544
def ichimoku(df):
    # Tenkan Sen
    tenkan_max = df['High'].rolling(window=9, min_periods=0).max()
    tenkan_min = df['Low'].rolling(window=9, min_periods=0).min()
    df['tenkan_avg'] = (tenkan_max + tenkan_min) / 2

    # Kijun Sen
    kijun_max = df['High'].rolling(window=26, min_periods=0).max()
    kijun_min = df['Low'].rolling(window=26, min_periods=0).min()
    df['kijun_avg'] = (kijun_max + kijun_min) / 2

    # Senkou Span A
    # (Kijun + Tenkan) / 2 Shifted ahead by 26 periods
    df['senkou_a'] = ((df['kijun_avg'] + df['tenkan_avg']) / 2).shift(26)

    # Senkou Span B
    # 52 period High + Low / 2
    senkou_b_max = df['High'].rolling(window=52, min_periods=0).max()
    senkou_b_min = df['Low'].rolling(window=52, min_periods=0).min()
    df['senkou_b'] = ((senkou_b_max + senkou_b_min) / 2).shift(52)

    # Chikou Span
    # Current close shifted -26
    df['chikou'] = (df['Close']).shift(-26)

from load_binance import BinanceData
import json

# Offline Loading
sig = "linkupusdt" + "@" + "1h" + "-" + str(1000) + "|" + "" + "-" + ""
#BinanceData.store[sig] = json.load(open("linkupusdt-1h-1000.json"))
data = BinanceData.loadKlineFrame(symbol="linkupusdt", timeframe="5m", startTime=1609459200000, endTime=1613055600000)
rsi, k, d = StochRSI(data.Close)
print(data)
data["K"] = k
data["D"] = d

ichimoku(data)
data.tail(2)
import matplotlib.pyplot as plt


#import plotly.graph_objects as go
#fig = go.Figure(data=[go.Candlestick(x=data['OpenTime'],
#                open=data['Open'],
#                high=data['High'],
#                low=data['Low'],
#                close=data['Close'])])

#fig.show()

import mplfinance as mpf
data['Date'] = pd.to_datetime(data["OpenTime"], unit='ms')
data['Date'] = pd.DatetimeIndex(data['Date'])
data.index.name = 'Date'
data.set_index('Date', inplace=True)
#mpf.plot(data, type='candle', style='charles', volume=False, savefig='price.pdf')
mpf.plot(data, type='candle', style='charles', volume=False)

def theme(figure, axList):
    for ax in axList:
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white', which='both')
        ax.set_facecolor("#0e101f")
        plt.setp(ax.get_title(), color='white')
        legend = ax.get_legend()
        if legend:
            frame = legend.get_frame()
            if frame:
                frame.set_facecolor('#202446')
                frame.set_edgecolor('#2D3262')
            for text in legend.get_texts():
                text.set_color("white")
    figure.set_facecolor("#0e101f")

# Plotting Code: https://stackoverflow.com/a/54862983
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 9), gridspec_kw={'height_ratios': [4, 1]})

ax = axs[0]
ax.plot(data.index, data.Close, linewidth=2, color="#000000")
ax.fill_between(data.index, data.senkou_a, data.senkou_b, where=data.senkou_a >= data.senkou_b, color='#9EE493AA')
ax.fill_between(data.index, data.senkou_a, data.senkou_b, where=data.senkou_a < data.senkou_b, color='#F75590AA')
#ax.plot(data.index, data.tenkan_avg, color="#0000ff")
#ax.plot(data.index, data.kijun_avg, color="#ffd100")

ax = axs[1]
ax.plot(data.index, data.K,linewidth=2, color="#00a1ff")
ax.plot(data.index, data.D,linewidth=2, color="#caba00")
#theme(fig, axs)
plt.savefig('line_plot.pdf')
plt.show()




# RSI-Stochastic CrossOver; src: https://pythonforfinance.net/2017/10/10/stochastic-oscillator-trading-strategy-backtest-in-python/
crossDown = ((data['K'] < data['D']) & (data['K'].shift(1) > data['D'].shift(1)))
crossUp = ((data['K'] > data['D']) & (data['K'].shift(1) < data['D'].shift(1)))
# Cloud State
cloudIsUp = (data['Close'] < data['senkou_a']) & (data['Close'] < data['senkou_b'])
cloudIsDown = (data['Close'] > data['senkou_a']) & (data['Close'] > data['senkou_b'])
# Threshold
rsiUpThresh = data['D'] > 0.7
rsiDownThresh = data['D'] < 0.3
data['Long'] = cloudIsUp & crossUp & rsiDownThresh
data['Short'] = cloudIsDown & crossDown & rsiUpThresh
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(20, 9))
ax.plot(data.index, data.Close, linewidth=2, color="#000000")

ax.fill_between(data.index, data.senkou_a, data.senkou_b, where=data.senkou_a >= data.senkou_b, color='#9EE493AA')
ax.fill_between(data.index, data.senkou_a, data.senkou_b, where=data.senkou_a < data.senkou_b, color='#F75590AA')

dx = data[data['Long'] == True]
dy = data[data['Short'] == True]
for i in range(len(dx)):
    plt.axvline(dx.iloc[i].name, color="#9EE493", lw=1)
for i in range(len(dy)):
    plt.axvline(dy.iloc[i].name, color="#F75590", lw=1)

#theme(fig, [ax])
plt.savefig('line_plot2.pdf')

plt.show()

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
cloudD = ctrl.Antecedent(np.arange(0, 100, 1), 'cloud-state')
rsiV = ctrl.Antecedent(np.arange(0, 100, 1), 'rsi-value')
crossOver = ctrl.Antecedent(np.arange(0, 100, 1), 'cross-over')
position = ctrl.Consequent(np.arange(0, 100, 1), 'position', defuzzify_method='bisector')
#cloudD.automf(3)
rsiV.automf(3)
crossOver.automf(3)
position.automf(3)
import warnings

warnings.filterwarnings("ignore")
from skfuzzy.control.visualization import FuzzyVariableVisualizer


def viz(var, sim=None):
    fig, ax = FuzzyVariableVisualizer(var).view(sim=sim)
    #theme(fig, [ax])
    plt.savefig(var.label+'.pdf')
    fig.show()

cloudD['poor'] = fuzz.trimf(cloudD.universe, [0, 0, 50])
cloudD['average'] = fuzz.trimf(cloudD.universe, [0, 50, 100])
cloudD['good'] = fuzz.trimf(cloudD.universe, [50, 100, 100])
#cloudD['medio'] = fuzz.gaussmf(cloudD.universe, 25, 3)
#cloudD['alto']  = fuzz.pimf(cloudD.universe, 30, 45, 50, 75)
viz(cloudD)
viz(rsiV)
viz(crossOver)
viz(position)

rule1 = ctrl.Rule(rsiV['poor'] & crossOver['good'], position['good'])
rule2 = ctrl.Rule(rsiV['good'] & crossOver['poor'], position['poor'])
rule3 = ctrl.Rule(cloudD['good'] & crossOver['poor'], position['poor'])
rule4 = ctrl.Rule(cloudD['poor'] & crossOver['good'], position['good'])
rule5 = ctrl.Rule(cloudD['poor'] & rsiV['poor'] & crossOver['good'], position['good'])
rule6 = ctrl.Rule(cloudD['good'] & rsiV['good'] & crossOver['poor'], position['poor'])
rule7 = ctrl.Rule(cloudD['average'] & rsiV['average'] & crossOver['average'], position['average'])
rule8 = ctrl.Rule(cloudD['good'] & rsiV['good'] & crossOver['good'], position['average'])
rule9 = ctrl.Rule(cloudD['poor'] & rsiV['poor'] & crossOver['poor'], position['average'])

pos_ctrl1 = ctrl.ControlSystem([rule1, rule2, rule3,
                                rule4, rule5, rule6,
                                rule7, rule8, rule9])
from skfuzzy.control.visualization import ControlSystemVisualizer
import networkx as nx


# https://www.python-graph-gallery.com/321-custom-networkx-graph-appearance
def vizR(rule):
    graph = rule.graph
    fig, ax = plt.subplots(figsize=(16,16))
    pos = nx.spring_layout(graph, k=0.8, iterations=10)
    nx.draw(graph,pos=pos, with_labels=True, node_size=3000, node_color="#FFF689", node_shape="o", alpha=0.8, linewidths=1, width=4, edge_color="#3DB1F5", ax=ax)
    plt.savefig('rules' + str(rule.label) + '.pdf')

    #theme(fig, [ax])

vizR(rule7)
cloudD2 = ctrl.Antecedent(np.arange(0, 100, 1), 'cloud-state')
rsiV2 = ctrl.Antecedent(np.arange(0, 100, 1), 'rsi-value')
crossOver2 = ctrl.Antecedent(np.arange(0, 100, 1), 'cross-over')
position2 = ctrl.Consequent(np.arange(0, 100, 1), 'position')

#cloudD2.automf(5)
cloudD2['poor'] = fuzz.trimf(cloudD2.universe, [0, 0, 25])
cloudD2['mediocre'] = fuzz.trimf(cloudD2.universe, [0, 25, 50])
cloudD2['average'] = fuzz.trimf(cloudD2.universe, [25, 50, 75])
cloudD2['decent'] = fuzz.trimf(cloudD2.universe, [50, 75, 100])
cloudD2['good'] = fuzz.trimf(cloudD2.universe, [75, 100, 100])

rsiV2.automf(5)
crossOver2.automf(5)
position2.automf(5)
viz(cloudD2)
dC = cloudD2['poor'] | cloudD2['mediocre']
uC = cloudD2['decent'] | cloudD2['good']

r1 = ctrl.Rule(dC & rsiV2['poor'] & crossOver2['good'], position2['good'])
r2 = ctrl.Rule(dC & rsiV2['mediocre'] & crossOver2['good'], position2['decent'])
r3 = ctrl.Rule(dC & rsiV2['poor'] & crossOver2['mediocre'], position2['decent'])

r4 = ctrl.Rule(uC & rsiV2['good'] & crossOver2['poor'], position2['poor'])
r5 = ctrl.Rule(uC & rsiV2['decent'] & crossOver2['poor'], position2['mediocre'])
r6 = ctrl.Rule(uC & rsiV2['good'] & crossOver2['decent'], position2['mediocre'])

r7 = ctrl.Rule(cloudD2['average'] & rsiV2['average'] & crossOver2['average'], position2['average'])
r8 = ctrl.Rule(uC & rsiV2['good'] & crossOver2['average'], position2['average'])
r9 = ctrl.Rule(dC & rsiV2['average'] & crossOver2['mediocre'], position2['average'])
r10 = ctrl.Rule(uC & rsiV2['average'] & crossOver2['decent'], position2['average'])
r11 = ctrl.Rule(dC & rsiV2['poor'] & crossOver2['average'], position2['average'])

r12 = ctrl.Rule(cloudD2['average'] & rsiV2['good'] & crossOver2['poor'], position2['mediocre'])
r13 = ctrl.Rule(cloudD2['average'] & rsiV2['poor'] & crossOver2['good'], position2['decent'])

r14 = ctrl.Rule(dC & rsiV2['good'] & crossOver2['poor'], position2['mediocre'])
r15 = ctrl.Rule(uC & rsiV2['poor'] & crossOver2['good'], position2['decent'])
r16 = ctrl.Rule(crossOver2['average'], position2['average'])

pos_ctrl2 = ctrl.ControlSystem([r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16])
vizR(r8)
pos = ctrl.ControlSystemSimulation(pos_ctrl2)
pos.input['cloud-state'] = 90
pos.input['rsi-value'] = 40
pos.input['cross-over'] = 40
pos.compute()
print(pos.output['position'])
viz(position2, sim=pos)

def plotSerie(s):
    fig, ax = plt.subplots()
    s.plot(ax=ax, color='#FFF689')
    theme(fig, [ax])
    plt.show()


dK = (data.K - data.D)
dK = abs(dK)
crossV = (crossDown * -1 * dK) + (crossUp * dK)

crossV[crossV > 0.2] = 0.2
crossV[crossV < -0.2] = -0.2
crossV = crossV + 0.2
crossV = crossV * (1 / 0.4)

data["cross-over"] = crossV

plotSerie(crossV)
cdata = data.dropna()
cdata.head(3)
cdata["cloudIsUp"] = (cdata['Close'] < cdata['senkou_a']) & (cdata['Close'] < cdata['senkou_b'])
cdata["cloudIsDown"] = (cdata['Close'] > cdata['senkou_a']) & (cdata['Close'] > cdata['senkou_b'])
cdata["cloudState"] = 0.5
cdata["cloudState"] += cdata["cloudIsUp"] * 0.5
cdata["cloudState"] += cdata["cloudIsDown"] * -0.5
cdata.head(3)
res = [
    [],
    []
]
control_systems = [pos_ctrl1, pos_ctrl2]
c = [0, 0]
for i in range(len(cdata)):
    d = cdata.iloc[i]
    pos = None
    for idx, system in enumerate(control_systems):
        try:
            pos = ctrl.ControlSystemSimulation(system)
            pos.input['cloud-state'] = (1 - d['cloudState']) * 100
            pos.input['rsi-value'] = d['K'] * 100
            pos.input['cross-over'] = d['cross-over'] * 100
            pos.compute()
            res[idx].append(pos.output['position'])
            #cloudD.view(sim=pos)
        except:
            res[idx].append(50)
            c[idx] += 1
print("unmatched inputs system 1: ", c[0])
print("unmatched inputs system 2: ", c[1])
cdata["fuzz"] = res[0]
cdata["fuzz2"] = res[1]


def plotsurface():
    upsampled = np.linspace(0, 100, 21)
    x, y = np.meshgrid(upsampled, upsampled)
    z = np.zeros_like(x)
    sim = ctrl.ControlSystemSimulation(pos_ctrl1)

    # Loop through the system 21*21 times to collect the control surface
    for i in range(21):
        for j in range(21):
            sim.input['rsi-value'] = x[i, j]
            sim.input['cross-over'] = y[i, j]
            sim.compute()
            z[i, j] = sim.output['output']

    import matplotlib.pyplot as plt  # noqa: E402

    # Required for 3D plotting
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                           linewidth=0.4, antialiased=True)

    cset = ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
    cset = ax.contourf(x, y, z, zdir='x', offset=3, cmap='viridis', alpha=0.5)
    cset = ax.contourf(x, y, z, zdir='y', offset=3, cmap='viridis', alpha=0.5)

    ax.view_init(30, 200)


def plotPositions(var, u, d):
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(20, 9))
    ax.plot(cdata.index, cdata.Close, linewidth=2, color="#000000")

    ax.fill_between(cdata.index, cdata.senkou_a, cdata.senkou_b, where=cdata.senkou_a >= cdata.senkou_b,
                    color='#9EE493AA')
    ax.fill_between(cdata.index, cdata.senkou_a, cdata.senkou_b, where=cdata.senkou_a < cdata.senkou_b,
                    color='#F75590AA')

    dx = cdata[cdata[var] > u]
    dy = cdata[cdata[var] < d]
    for i in range(len(dx)):
        plt.axvline(dx.iloc[i].name, color="#9EE493", lw=1)
    for i in range(len(dy)):
        plt.axvline(dy.iloc[i].name, color="#F75590", lw=1)

    plt.savefig(str(var)+'position.pdf')
    #theme(fig, [ax])


plotPositions("fuzz", 77, 23)
plt.show()
plotPositions("fuzz2", 60, 40)

num_resolution = 31


def plotsurfaceRIS_CROSS():
    upsampled = np.linspace(0, 100, num_resolution)
    x, y = np.meshgrid(upsampled, upsampled)
    z = np.zeros_like(x)
    sim = ctrl.ControlSystemSimulation(pos_ctrl1)

    # Loop through the system 21*21 times to collect the control surface
    for i in range(num_resolution):
        for j in range(num_resolution):
            sim.input['rsi-value'] = x[i, j]
            sim.input['cross-over'] = y[i, j]
            sim.compute()
            z[i, j] = sim.output['position']

    import matplotlib.pyplot as plt  # noqa: E402

    # Required for 3D plotting
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                           linewidth=0.4, antialiased=True)

    ax.set_xlabel('RSI')
    ax.set_ylabel('CROSS')
    ax.set_zlabel('POSITION')

    ax.view_init(30, -45)
    plt.show()


def plotsurfaceRIS_CLOUD():
    upsampled = np.linspace(0, 100, num_resolution)
    x, y = np.meshgrid(upsampled, upsampled)
    z = np.zeros_like(x)
    sim = ctrl.ControlSystemSimulation(pos_ctrl1)

    # Loop through the system 21*21 times to collect the control surface
    for i in range(num_resolution):
        for j in range(num_resolution):
            sim.input['cloud-state'] = y[i, j]
            sim.input['rsi-value'] = x[i, j]
            sim.compute()
            z[i, j] = sim.output['position']

    import matplotlib.pyplot as plt  # noqa: E402

    # Required for 3D plotting
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                           linewidth=0.4, antialiased=True)

    ax.set_xlabel('RSI')
    ax.set_ylabel('CLOUD')
    ax.set_zlabel('POSITION')

    ax.view_init(30, 45)
    plt.show()


def plotsurfaceCROSS_CLOUD():
    upsampled = np.linspace(0, 100, num_resolution)
    x, y = np.meshgrid(upsampled, upsampled)
    z = np.zeros_like(x)
    sim = ctrl.ControlSystemSimulation(pos_ctrl1)

    # Loop through the system 21*21 times to collect the control surface
    for i in range(num_resolution):
        for j in range(num_resolution):
            sim.input['cloud-state'] = y[i, j]
            sim.input['cross-over'] = x[i, j]
            sim.compute()
            z[i, j] = sim.output['position']

    import matplotlib.pyplot as plt  # noqa: E402

    # Required for 3D plotting
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                           linewidth=0.4, antialiased=True)

    ax.set_xlabel('CROSS')
    ax.set_ylabel('CLOUD')
    ax.set_zlabel('POSITION')

    ax.view_init(30, 200)
    plt.show()


#plotsurfaceRIS_CROSS()
#plotsurfaceRIS_CLOUD()
#plotsurfaceCROSS_CLOUD()



# Eval Df for profits
def evalProfit(var, upT, downT):
    position = None
    positions = []
    profit = 1

    maxLoss = -0.2

    for i in range(len(cdata)):
        x = cdata.iloc[i]
        f = x[var]
        if f < downT and position is None:
            position = {
                "type": "short",
                "enter": x.Close,
                "check": x.High * 1.07,
                "start_date": cdata.index[i]
            }
        elif f > upT and position is None:
            position = {
                "type": "long",
                "enter": x.Close,
                "check": x.Low * 0.93,
                "start_date": cdata.index[i]
            }
        elif position is not None:
            if position["type"] == "short":
                r = (position["enter"] - x.Close) / position["enter"]
                if f > upT:
                    position["exit"] = x.Close
                    position["end_date"] = cdata.index[i]

                    positions.append(position)
                    profit *= (1 + r)
                    position = None
                elif r < maxLoss:
                    position["exit"] = position["enter"] * 1.1
                    position["end_date"] = cdata.index[i]
                    r = maxLoss
                    positions.append(position)
                    profit *= (1 + r)
                    position = None
            elif position["type"] == "long":
                r = (x.Close - position["enter"]) / position["enter"]
                if f < downT:
                    position["end_date"] = cdata.index[i]
                    position["exit"] = x.Close
                    positions.append(position)
                    profit *= (1 + r)
                    position = None
                elif r < maxLoss:
                    position["end_date"] = cdata.index[i]
                    position["exit"] = position["enter"] * 0.9
                    r = maxLoss
                    positions.append(position)
                    profit *= (1 + r)
                    position = None
    return profit, positions

res1, positions1 = evalProfit("fuzz", 77, 23)
res2, positions2 = evalProfit("fuzz2", 60, 40)



def plotPositions2(var, positions):
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(20, 9))
    ax.plot(cdata.index, cdata.Close, linewidth=2, color="#000000")

    ax.fill_between(cdata.index, cdata.senkou_a, cdata.senkou_b, where=cdata.senkou_a >= cdata.senkou_b,
                    color='#9EE493AA')
    ax.fill_between(cdata.index, cdata.senkou_a, cdata.senkou_b, where=cdata.senkou_a < cdata.senkou_b,
                    color='#F75590AA')

    for i in range(len(positions)):
        if positions[i]['type'] == 'short':
            ax.axvspan(positions[i]['start_date'], positions[i]['end_date'], alpha=0.2, color='red')
        else:
            ax.axvspan(positions[i]['start_date'], positions[i]['end_date'], alpha=0.2, color='green')
            # ax.axvspan(positions[i]['enter'], positions[i]['exit'], alpha=0.5, color='green')


    plt.savefig(str(var)+'res_position.pdf')
    #theme(fig, [ax])


plotPositions2("fuzz", positions1)
plt.show()

plotPositions2("fuzz2", positions2)
plt.show()

print("res1 = " + str(res1) + " in " + str(len(positions1)) + " positions")
print("res2 = " + str(res2) + " in " + str(len(positions2)) + " positions")
