Reinforcement Learning for Stock Market Trading
Strategy using Q_Learning
This report details the work of our group on developing a stock market trading strategy using QLearning. The objective was to train a Reinforcement Learning (RL) agent to learn an optimal trading
policy using historical price data and technical indicators. The chosen trading style was not explicitly
mentioned.


1. Objective
The primary objective of this project was to develop a trading strategy using Reinforcement
Learning, specifically by training an RL agent to learn an optimal trading policy. This involved using
historical price data and technical indicators as inputs.


2. Data Collection & Preprocessing
  The project utilized historical price data for Apple Inc. (AAPL) obtained from the Yahoo Finance
  API (yfinance). The timeframe was configurable, with the option for daily data. The data acquisition
  was handled by the load_data function.

  1) Technical Indicators:
    Several technical indicators were implemented, including:
  2) Trend Indicators:
    SMA (Simple Moving Average) with periods of 20 and 50
  3) Momentum Indicators:
    RSI (Relative Strength Index), which provides overbought/oversold signals.
  4) Data Normalization & Splitting:
    The numeric features were scaled between 0 and 1 for model efficiency through data normalisation.
    The data was split temporally into 80% for training and 20% for testing, handled by the
    split_data function.



3. RL Algorithm Implementation – Q-Learning
  The project implemented Tabular Q-Learning. Key components included:
  A Q-table to store state-action values, implemented as a nested defaultdict.
  An epsilon-greedy policy for balancing exploration and exploitation.
  Learning rate (α), discount factor (γ), and exploration rate (ε). The learning rate was set to
  0.001 and the discount factor to 0.95.
  Q-value updates based on the Bellman Equation:
  Q(s,a) = Q(s,a) + α[r + γ * max Q(s',a') - Q(s,a)] .

 ** Implementation Details**

  The Q-table maps discretized states to action values, with a default value of 0 for new state-action
  pairs.
  Continuous states were discretized for Q-table use.
  Action selection followed an epsilon-greedy policy:
  With probability ε, a random action is chosen (exploration).
  With probability 1 - ε, the best-known action is chosen (exploitation).
  The exploration rate (ε) was gradually reduced over time (exploration rate decay), starting at 1.0
  and decreasing to approximately 0.6 over 50 episodes.



4. Training and Evaluation
The dataset was split into training and testing sets. The agent learned by maximizing rewards
across multiple episodes. The goal was to optimise the agent's policy through repeated interactions
with the environment.

**Evaluation Metrics**
The following metrics were calculated to evaluate the model's performance:

1. Total Profit: Overall portfolio gain or loss.
2. Sharpe Ratio: Risk-adjusted return measurement.
3. Max Drawdown: Largest drop in portfolio value.
4. Win Rate: Percentage of profitable trades.
5. Cumulative Return: Total return over time.
