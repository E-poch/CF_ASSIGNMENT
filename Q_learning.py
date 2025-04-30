
import os
import numpy as np
import pandas as pd
import yfinance as yf
import logging
import pickle
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class TechnicalIndicators:
    def __init__(self, data):
        self.data = data.copy()

    def add_sma(self, window=20):
        self.data[f'sma_{window}'] = self.data['close'].rolling(window=window).mean()
        return self.data

    def add_rsi(self, window=14):
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        self.data[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        self.data[f'rsi_{window}'].replace([np.inf, -np.inf], 100, inplace=True)
        self.data[f'rsi_{window}'].fillna(50, inplace=True)
        return self.data

    def add_all_indicators(self):
        self.add_sma(window=20)
        self.add_sma(window=50)
        self.add_rsi(window=14)
        self.data.dropna(inplace=True)
        return self.data


class TradingEnvironment:
    def __init__(self, data, initial_balance=10000, transaction_fee_percent=0.001, window_size=10):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.window_size = window_size
        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()
        self.current_step = 0
        self.balance = 0
        self.shares_held = 0
        self.net_worth = 0
        self.trades = []

    def _create_action_space(self):
        class ActionSpace:
            def __init__(self):
                self.n = 3

        return ActionSpace()

    def _create_observation_space(self):
        num_features = len(self._get_relevant_columns())
        shape = (self.window_size * num_features,)

        class ObservationSpace:
            def __init__(self, shape):
                self.shape = shape

        return ObservationSpace(shape)

    def _get_relevant_columns(self):
        cols = ['close']
        indicator_cols = [c for c in self.data.columns if c.startswith('sma_') or c.startswith('rsi_')]
        cols.extend(indicator_cols)
        existing_cols = [c for c in cols if c in self.data.columns]
        if not existing_cols:
            raise ValueError("no relevant columns found for state.")
        return existing_cols

    def _get_state(self):
        if self.current_step < self.window_size - 1:
            num_features = len(self._get_relevant_columns())
            return np.zeros(self.window_size * num_features)
        start = self.current_step - self.window_size + 1
        end = self.current_step + 1
        relevant_cols = self._get_relevant_columns()
        state_data = self.data[relevant_cols].iloc[start:end].values
        return state_data.flatten()

    def reset(self):
        self.current_step = self.window_size - 1
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.trades = []
        state = self._get_state()
        info = {'portfolio_value': self.net_worth, 'balance': self.balance, 'shares_held': self.shares_held, 'total_trades': 0, 'current_step': self.current_step}
        return state, info

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        truncated = False
        current_price = self.data['close'].iloc[self.current_step]
        trade_executed = False
        trade_type = 'hold'
        trade_shares = 0
        trade_price = 0

        if action == 1:
            if self.balance > 0:
                shares_to_buy = self.balance / current_price
                fee = shares_to_buy * current_price * self.transaction_fee_percent
                if self.balance >= shares_to_buy * current_price + fee:
                    self.shares_held += shares_to_buy
                    self.balance -= (shares_to_buy * current_price + fee)
                    trade_executed = True
                    trade_type = 'buy'
                    trade_shares = shares_to_buy
                    trade_price = current_price
        elif action == 2:
            if self.shares_held > 0:
                sell_value = self.shares_held * current_price
                fee = sell_value * self.transaction_fee_percent
                self.balance += sell_value - fee
                trade_shares = self.shares_held
                self.shares_held = 0
                trade_executed = True
                trade_type = 'sell'
                trade_price = current_price

        prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * current_price
        reward = self.net_worth - prev_net_worth

        if trade_executed:
            self.trades.append({
                'step': self.current_step,
                'type': trade_type,
                'price': trade_price,
                'shares': trade_shares,
                'portfolio_value': self.net_worth
            })

        next_state = self._get_state() if not done else np.zeros_like(self._get_state())
        info = {
            'portfolio_value': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'total_trades': len(self.trades),
            'current_step': self.current_step
        }

        return next_state, reward, done, truncated, info

    def get_trade_history(self):
        return pd.DataFrame(self.trades)


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0,
                 exploration_decay=0.995, min_exploration_rate=0.01, state_discretization_bins=10):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.state_discretization_bins = state_discretization_bins
        self.action_size = env.action_space.n
        self.state_size = env.observation_space.shape[0]
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))
        self.rewards_history = []
        self.portfolio_values = []
        self.exploration_rates = []
        self.state_low_bounds = np.full(self.state_size, -1.0)
        self.state_high_bounds = np.full(self.state_size, 1.0)

    def _discretize_state(self, state):
        state = np.array(state).flatten()
        if len(state) != self.state_size:
            return tuple(np.zeros(self.state_size, dtype=int))
        discrete_state = []
        for i in range(self.state_size):
            scaled_value = max(0.0, min(1.0, (state[i] - self.state_low_bounds[i]) / (self.state_high_bounds[i] - self.state_high_bounds[i])))
            bin_index = int(scaled_value * (self.state_discretization_bins - 1))
            discrete_state.append(bin_index)
        return tuple(discrete_state)

    def select_action(self, state, use_exploration=True):
        discrete_state = self._discretize_state(state)
        if use_exploration and random.uniform(0, 1) < self.exploration_rate:
            return random.randrange(self.action_size)
        else:
            q_values = self.q_table.get(discrete_state, np.zeros(self.action_size))
            return np.argmax(q_values)

    def learn(self, state, action, reward, next_state, done):
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)
        current_q = self.q_table[discrete_state][action]
        next_max_q = np.max(self.q_table.get(discrete_next_state, np.zeros(self.action_size)))
        new_q = reward if done else current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[discrete_state][action] = new_q

    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def train(self, num_episodes=1000, max_steps=None):
        history = {'rewards': [], 'portfolio_values': [], 'exploration_rates': []}
        for episode in range(num_episodes):
            state, info = self.env.reset()
            state = self._normalize_state(state)
            done = False
            truncated = False
            episode_reward = 0
            step = 0
            while not (done or truncated):
                action = self.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = self._normalize_state(next_state)
                self.learn(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                step += 1
                if max_steps is not None and step >= max_steps:
                    truncated = True
                    break
            self.decay_exploration()
            history['rewards'].append(episode_reward)
            history['portfolio_values'].append(info['portfolio_value'])
            history['exploration_rates'].append(self.exploration_rate)
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(history['rewards'][-10:])
                avg_portfolio = np.mean(history['portfolio_values'][-10:])
                print(f"episode {episode + 1}/{num_episodes} - avg reward (last 10): {avg_reward:.2f}, avg portfolio (last 10): {avg_portfolio:.2f}, exploration: {self.exploration_rate:.4f}")
        self.rewards_history = history['rewards']
        self.portfolio_values = history['portfolio_values']
        self.exploration_rates = history['exploration_rates']
        return history

    def test(self, test_env=None, render=False, test_exploration_rate=0.0):
        env = test_env if test_env else self.env
        state, info = env.reset()
        state = self._normalize_state(state)
        done = False
        truncated = False
        total_reward = 0
        initial_portfolio = info['portfolio_value']
        portfolio_values = [initial_portfolio]
        actions_taken = []
        original_exploration_rate = self.exploration_rate
        self.exploration_rate = test_exploration_rate
        while not (done or truncated):
            action = self.select_action(state, use_exploration=True)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = self._normalize_state(next_state)
            state = next_state
            total_reward += reward
            portfolio_values.append(info['portfolio_value'])
            actions_taken.append(action)
            if render:
                env.render()
        self.exploration_rate = original_exploration_rate
        final_portfolio = portfolio_values[-1]
        final_return = (final_portfolio / initial_portfolio) - 1 if initial_portfolio > 0 else 0
        trades = env.get_trade_history()
        results = {
            'total_reward': total_reward,
            'portfolio_values': portfolio_values,
            'actions': actions_taken,
            'return': final_return,
            'trades': trades,
            'info': info
        }
        return results

    def _normalize_state(self, state):
        state = np.array(state).flatten()
        if len(state) != self.state_size:
            return np.zeros(self.state_size)
        normalized_state = (state - self.state_low_bounds) / (self.state_high_bounds - self.state_low_bounds)
        normalized_state = np.clip(normalized_state, 0, 1)
        return normalized_state

    def save(self, filepath):
        if os.path.isdir(filepath):
            filepath = os.path.join(filepath, 'q_learning_model.pkl')
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        q_table_dict = dict(self.q_table)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': q_table_dict,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'exploration_rate': self.exploration_rate,
                'exploration_decay': self.exploration_decay,
                'min_exploration_rate': self.min_exploration_rate,
                'state_discretization_bins': self.state_discretization_bins,
                'state_low_bounds': self.state_low_bounds,
                'state_high_bounds': self.state_high_bounds,
                'rewards_history': self.rewards_history,
                'portfolio_values': self.portfolio_values,
                'exploration_rates': self.exploration_rates
            }, f)

    @classmethod
    def load(cls, filepath, env):
        if os.path.isdir(filepath):
            filepath = os.path.join(filepath, 'q_learning_model.pkl')
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        agent = cls(env,
                    learning_rate=data['learning_rate'],
                    discount_factor=data['discount_factor'],
                    exploration_rate=data['exploration_rate'],
                    exploration_decay=data['exploration_decay'],
                    min_exploration_rate=data['min_exploration_rate'],
                    state_discretization_bins=data['state_discretization_bins'])
        agent.q_table = defaultdict(lambda: np.zeros(agent.action_size))
        agent.q_table.update(data['q_table'])
        agent.state_low_bounds = data['state_low_bounds']
        agent.state_high_bounds = data['state_high_bounds']
        agent.rewards_history = data.get('rewards_history', [])
        agent.portfolio_values = data.get('portfolio_values', [])
        agent.exploration_rates = data.get('exploration_rates', [])
        return agent


def load_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"no data found for {symbol} between {start_date} and {end_date}")
        data.reset_index(inplace=True)
        if 'date' not in data.columns and 'Date' not in data.columns:
            if 'index' in data.columns:
                data['date'] = pd.to_datetime(data['index'])
            else:
                raise ValueError("no 'date' or 'Date' or 'index' column found in downloaded data.")
        elif 'date' not in data.columns:
            data.rename(columns={'Date': 'date'}, inplace=True)
            data['date'] = pd.to_datetime(data['date'])
        else:
            data['date'] = pd.to_datetime(data['date'])

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in data.columns:
                col_title = col.title()
                if col_title in data.columns:
                    data.rename(columns={col_title: col}, inplace=True)
                else:
                    raise ValueError(f"required column '{col}' not found in downloaded data.")
        return data
    except Exception as e:
        raise ValueError(f"error loading data: {e}")


def split_data(data, test_size=0.2):
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    return train_data, test_data

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])


def calculate_buy_and_hold(data, initial_balance, window_size):
    buy_and_hold = [initial_balance]
    initial_price = data['close'].iloc[window_size]
    for i in range(1, len(data) - window_size):
        current_price = data['close'].iloc[window_size + i]
        buy_and_hold_value = initial_balance * (current_price / initial_price)
        buy_and_hold.append(buy_and_hold_value)
    return buy_and_hold

def calculate_ma_crossover(data, initial_balance, window_size):
    ma_crossover = [initial_balance]
    short_window = 20
    long_window = 50
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0
    signals['short_mavg'] = data['close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = data['close'].rolling(window=long_window, min_periods=1, center=False).mean()
    signals['signal'][window_size:] = np.where(signals['short_mavg'][window_size:] > signals['long_mavg'][window_size:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()

    balance = initial_balance
    shares = 0
    for i in range(1, len(data) - window_size):
        if signals['positions'][window_size + i] == 1:
            shares = balance / data['close'].iloc[window_size + i]
            balance = 0
        elif signals['positions'][window_size + i] == -1:
            balance = shares * data['close'].iloc[window_size + i]
            shares = 0
        ma_crossover.append(balance + shares * data['close'].iloc[window_size + i])
    return ma_crossover

def calculate_total_return(portfolio_values, initial_balance):
    return (portfolio_values[-1] / initial_balance) - 1

def calculate_sharpe_ratio(returns, risk_free_rate=0):
    returns = pd.Series(returns)
    if len(returns) < 2:
        return 0
    excess_returns = returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe_ratio

def calculate_max_drawdown(portfolio_values):
    peak = portfolio_values[0]
    max_drawdown = 0
    for value in portfolio_values:
        peak = max(peak, value)
        drawdown = (peak - value) / peak if peak != 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def plot_results(data, results, output_dir, initial_balance, window_size):
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    dates = pd.to_datetime(data['date'])
    dates_mpl = mdates.date2num(dates)

    price = data['close']
    sma20 = data['sma_20']
    sma50 = data['sma_50']

    axs[0].plot(dates_mpl, price, label='price', linewidth=1.5)
    axs[0].plot(dates_mpl, sma20, label='sma 20', linestyle='--')
    axs[0].plot(dates_mpl, sma50, label='sma 50', linestyle='--')

    trades_df = results.get('trades')
    if trades_df is not None and not trades_df.empty:
        buy_trades = trades_df[trades_df['type'] == 'buy']
        sell_trades = trades_df[trades_df['type'] == 'sell']

        if not buy_trades.empty:
            buy_indices = buy_trades['step'].values
            valid_buy_indices = buy_indices < len(dates)
            if np.any(valid_buy_indices):
                buy_indices = buy_indices[valid_buy_indices]
                buy_dates_mpl = dates_mpl[buy_indices]
                axs[0].scatter(buy_dates_mpl, price.iloc[buy_indices], color='red', marker='v', label='buy', zorder=5)

        if not sell_trades.empty:
            sell_indices = sell_trades['step'].values
            valid_sell_indices = sell_indices < len(dates)
            if np.any(valid_sell_indices):
                sell_indices = sell_indices[valid_sell_indices]
                sell_dates_mpl = dates_mpl[sell_indices]
                axs[0].scatter(sell_dates_mpl, price.iloc[sell_indices], color='green', marker='^', label='sell', zorder=5)

    axs[0].set_ylabel("price ($)")
    axs[0].set_title("rl and sma performance trends")
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0].legend(by_label.values(), by_label.keys())
    axs[0].grid(True)


    rl_portfolio = results['portfolio_values']
    steps = range(len(rl_portfolio))

    ma_portfolio = calculate_ma_crossover(data.reset_index(drop=True), initial_balance, window_size)

    axs[1].plot(steps, rl_portfolio, label='rl strategy')
    axs[1].plot(steps, ma_portfolio[:len(steps)], '--', label='ma strategy')


    axs[1].set_ylabel("portfolio value ($)")
    axs[1].set_title("equity curve comparison")
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_xlabel("trading steps (test period)")


    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    axs[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(axs[1].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plot_filename = os.path.join(output_dir, 'rl_vs_ma_comparison.png')
    plt.savefig(plot_filename)
    plt.close(fig)


def main(symbol='aapl', start_date='2018-01-01', end_date='2023-01-01', test_size=0.2,
         initial_balance=10000, transaction_fee=0.001, window_size=10, episodes=100,
         learning_rate=0.001, discount_factor=0.95, exploration_rate=1.0,
         exploration_decay=0.995, min_exploration_rate=0.01, state_bins=10,
         output_dir='results', load_model=False, plot_results_flag=True):

    setup_logging(output_dir)

    try:
        data = load_data(symbol, start_date, end_date)
        indicators = TechnicalIndicators(data)
        indicators.add_sma(window=20)
        indicators.add_sma(window=50)
        indicators.add_rsi(window=14)
        data_with_indicators = indicators.data.dropna().copy()
        train_data, test_data = split_data(data_with_indicators, test_size=test_size)
    except ValueError as e:
        logging.error(e)
        return

    train_env = TradingEnvironment(train_data, initial_balance, transaction_fee, window_size)
    test_env = TradingEnvironment(test_data, initial_balance, transaction_fee, window_size)

    agent = QLearningAgent(train_env, learning_rate, discount_factor, exploration_rate,
                           exploration_decay, min_exploration_rate, state_bins)

    model_path = os.path.join(output_dir, 'q_learning_model.pkl')

    if load_model and os.path.exists(model_path):
        agent = QLearningAgent.load(model_path, train_env)
        logging.info(f"loaded model from {model_path}")
    else:
        history = agent.train(num_episodes=episodes)
        agent.save(model_path)
        logging.info(f"trained and saved model to {model_path}")

    results = agent.test(test_env)

    if plot_results_flag:
        if 'date' not in test_data.columns and test_data.index.name == 'date':
             test_data = test_data.reset_index()
        elif 'date' not in test_data.columns:
             logging.warning("could not find 'date' column in test_data for plotting x-axis.")
             try:
                 test_data = test_data.reset_index()
                 if 'date' not in test_data.columns and 'index' in test_data.columns:
                     test_data.rename(columns={'index': 'date'}, inplace=True)
                 if 'date' in test_data.columns:
                     test_data['date'] = pd.to_datetime(test_data['date'])
             except Exception as plot_data_err:
                 logging.error(f"error preparing test_data for plotting: {plot_data_err}")
                 return

        if 'sma_20' not in test_data.columns or 'sma_50' not in test_data.columns:
             logging.warning("sma_20 or sma_50 not found in test_data. re-calculating for plot.")
             temp_indicators = TechnicalIndicators(test_data)
             test_data['sma_20'] = temp_indicators.data['close'].rolling(window=20).mean()
             test_data['sma_50'] = temp_indicators.data['close'].rolling(window=50).mean()
             test_data.dropna(subset=['sma_20', 'sma_50'], inplace=True)


        plot_results(test_data.reset_index(drop=True), results, output_dir, initial_balance, window_size)
        logging.info(f"results plot saved to {os.path.join(output_dir, 'rl_vs_ma_comparison.png')}")


if __name__ == "__main__":
    try:
        main(
            symbol='aapl',
            start_date='2020-01-01',
            end_date='2023-01-01',
            test_size=0.2,
            initial_balance=10000,
            transaction_fee=0.001,
            window_size=10,
            episodes=100,
            learning_rate=0.001,
            discount_factor=0.95,
            exploration_rate=1.0,
            exploration_decay=0.995,
            min_exploration_rate=0.01,
            state_bins=10,
            output_dir='results',
            load_model=False,
            plot_results_flag=True
        )
    except ValueError as e:
        print(f"error during execution: {e}")
