"""
MarketEnvironment
=================
A decentralised double-auction market.  Agents hold inventories of two
goods (A, B) and can post buy/sell offers each step.  A lightweight
clearing mechanism matches compatible orders.  Price discovery and
specialisation emerge from agent behaviour.

State  : [inv_A, inv_B, price_A, price_B, cash, step_frac]  (per agent)
Action : [offer_type, good, quantity_frac, price_frac]
         offer_type  : 0 = hold, 1 = buy, 2 = sell
         good        : 0 = A, 1 = B
         quantity    : fraction of inventory to trade  in [0,1]
         price       : relative price offer  in [0.5, 2.0]
Reward : change in portfolio value (mark-to-market)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import gym
from gym import spaces


class Order:
    __slots__ = ("agent_id", "side", "good", "qty", "price")

    def __init__(self, agent_id, side, good, qty, price):
        self.agent_id = agent_id
        self.side = side   # "buy" | "sell"
        self.good = good   # 0 | 1
        self.qty = qty
        self.price = price


class MarketEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        n_agents: int = 6,
        init_inv: float = 50.0,
        init_cash: float = 200.0,
        max_steps: int = 300,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.init_inv = init_inv
        self.init_cash = init_cash
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # State per agent: [inv_A/K, inv_B/K, p_A/p_max, p_B/p_max,
        #                    cash/cash_max, step_frac]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        # [offer_type (3), good (2), qty_frac (1), price_frac (1)] → 4 dims
        # Discretised: offer_type ∈ {0,1,2}, good ∈ {0,1} encoded as floats
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0.0, 0.0], dtype=np.float32),
            high=np.array([2, 1, 1.0, 1.0], dtype=np.float32),
        )

        self.inv: Optional[np.ndarray] = None   # (n_agents, 2)
        self.cash: Optional[np.ndarray] = None  # (n_agents,)
        self.prices = np.array([1.0, 1.0])      # reference price per good
        self._step_count = 0
        self._trade_log: List[dict] = []

    # ------------------------------------------------------------------

    def reset(self) -> List[np.ndarray]:
        # Heterogeneous endowments: some agents are A-rich, others B-rich
        self.inv = np.full((self.n_agents, 2), self.init_inv, dtype=np.float32)
        for i in range(self.n_agents):
            good = i % 2
            self.inv[i, good] *= 2.0
            self.inv[i, 1 - good] *= 0.5
        self.cash = np.full(self.n_agents, self.init_cash, dtype=np.float32)
        self.prices = np.array([1.0, 1.0], dtype=np.float32)
        self._step_count = 0
        self._trade_log = []
        return self._get_obs()

    def step(
        self, actions: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[float], bool, dict]:
        orders = self._parse_actions(actions)
        trades = self._clear_market(orders)
        self._execute_trades(trades)
        self._update_prices(trades)

        rewards = self._compute_rewards()
        self._step_count += 1
        done = self._step_count >= self.max_steps

        info = {
            "prices": self.prices.copy(),
            "n_trades": len(trades),
            "total_volume": sum(t["qty"] for t in trades),
        }
        return self._get_obs(), rewards, done, info

    # ------------------------------------------------------------------
    # Market mechanics
    # ------------------------------------------------------------------

    def _parse_actions(self, actions: List[np.ndarray]) -> List[Order]:
        orders = []
        for i, a in enumerate(actions):
            offer_type = int(np.clip(round(float(a[0])), 0, 2))
            good = int(np.clip(round(float(a[1])), 0, 1))
            qty_frac = float(np.clip(a[2], 0, 1))
            price_frac = float(np.clip(a[3], 0, 1))

            if offer_type == 0:
                continue

            max_qty = (
                self.inv[i, good] if offer_type == 2 else self.cash[i] / (self.prices[good] + 1e-6)
            )
            qty = qty_frac * max_qty
            # Price ∈ [0.5 * ref, 2.0 * ref]
            price = self.prices[good] * (0.5 + 1.5 * price_frac)

            if qty > 1e-4:
                side = "sell" if offer_type == 2 else "buy"
                orders.append(Order(i, side, good, qty, price))
        return orders

    def _clear_market(self, orders: List[Order]) -> List[dict]:
        """Simple price-time-priority matching."""
        trades = []
        for good in range(2):
            buys = sorted(
                [o for o in orders if o.good == good and o.side == "buy"],
                key=lambda o: -o.price,
            )
            sells = sorted(
                [o for o in orders if o.good == good and o.side == "sell"],
                key=lambda o: o.price,
            )
            bi, si = 0, 0
            while bi < len(buys) and si < len(sells):
                b, s = buys[bi], sells[si]
                if b.agent_id == s.agent_id or b.price < s.price:
                    bi += 1
                    continue
                clear_price = (b.price + s.price) / 2.0
                qty = min(b.qty, s.qty)
                trades.append(
                    {
                        "buyer": b.agent_id,
                        "seller": s.agent_id,
                        "good": good,
                        "qty": qty,
                        "price": clear_price,
                    }
                )
                b.qty -= qty
                s.qty -= qty
                if b.qty < 1e-6:
                    bi += 1
                if s.qty < 1e-6:
                    si += 1
        return trades

    def _execute_trades(self, trades: List[dict]):
        for t in trades:
            b, s, g, q, p = t["buyer"], t["seller"], t["good"], t["qty"], t["price"]
            cost = q * p
            if self.cash[b] >= cost and self.inv[s, g] >= q:
                self.inv[b, g] += q
                self.inv[s, g] -= q
                self.cash[b] -= cost
                self.cash[s] += cost
        self._trade_log.extend(trades)

    def _update_prices(self, trades: List[dict]):
        for good in range(2):
            good_trades = [t for t in trades if t["good"] == good]
            if good_trades:
                vwap = sum(t["qty"] * t["price"] for t in good_trades) / (
                    sum(t["qty"] for t in good_trades) + 1e-8
                )
                self.prices[good] = 0.9 * self.prices[good] + 0.1 * vwap

    def _compute_rewards(self) -> List[float]:
        port_val = self.cash + self.inv @ self.prices
        prev_val = self.init_cash + self.init_inv * self.prices.sum()
        return ((port_val - prev_val) / (prev_val + 1e-8)).tolist()

    def _get_obs(self) -> List[np.ndarray]:
        inv_max = self.init_inv * 3
        cash_max = self.init_cash * 5
        p_max = 5.0
        obs = []
        for i in range(self.n_agents):
            o = np.array(
                [
                    self.inv[i, 0] / inv_max,
                    self.inv[i, 1] / inv_max,
                    self.prices[0] / p_max,
                    self.prices[1] / p_max,
                    self.cash[i] / cash_max,
                    self._step_count / self.max_steps,
                ],
                dtype=np.float32,
            )
            obs.append(o)
        return obs

    def render(self, mode="human"):
        total_wealth = (self.cash + self.inv @ self.prices).sum()
        print(
            f"Step {self._step_count:3d} | Prices A={self.prices[0]:.3f} B={self.prices[1]:.3f}"
            f" | Total wealth: {total_wealth:.1f}"
        )

    def market_efficiency(self) -> float:
        """
        Ratio of realised gains from trade vs theoretical maximum.
        Proxy: volume-weighted price spread narrowing over episode.
        """
        if len(self._trade_log) < 2:
            return 0.0
        early = np.mean([t["price"] for t in self._trade_log[:5]])
        late = np.mean([t["price"] for t in self._trade_log[-5:]])
        # Prices converging toward equilibrium ⟹ efficiency ↑
        return float(1.0 - abs(early - late) / (early + 1e-8))
