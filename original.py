import argparse
from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '7.5'

HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365
GAS_OPEX = 5
TIER_BENCHMARK = 0.37 #TODO change
URANIUM_PRICE = 0.94  # CAD/GJ
URANIUM_CONVERSION = 10.8
TIME = 40 * DAYS_PER_YEAR
ANNUAL_DISCOUNT_RATE = 0.05
DISCOUNT_RATE = (1 + ANNUAL_DISCOUNT_RATE) ** (1 / DAYS_PER_YEAR) - 1

CCGT_BUILD_COST = 1.5e9
CCGT_CAPACITY_FACTOR = 0.60
CCGT_CAPACITY = 1000
CCGT_FIXED_OM = 40 * 10 ** 6 / DAYS_PER_YEAR
CCGT_OPEX = 5

NUC_BUILD_COST = 8e9
NUC_CAPACITY_FACTOR = 0.9
NUC_CAPACITY = 1000
NUC_FIXED_OM = 150 * 1.37 * 10 ** 6 / DAYS_PER_YEAR
NUC_OPEX = 30

DISCOUNTS = np.array([1 / ((1 + DISCOUNT_RATE) ** t) for t in range(TIME)])
WINDOW = 180


@dataclass
class GasPriceConfig:
    starting_price: float = 1.5 * 1 #TODO change
    average_price: float = 1.5 * 1 #TODO change and save
    std: float = 0.5
    mean_reversion: float = 0.1


@dataclass
class CarbonPriceConfig:
    increase_days: int = 4 * DAYS_PER_YEAR
    start: int = 95
    end: int = 170


@dataclass
class ElectricityPriceConfig:
    scarcity_prob: float = 0.05
    scarcity_price: float = 500
    t_on_margin_low_eff_gas: float = 0.68
    t_on_margin_high_eff_gas: float = 0.21
    t_on_margin_zero_price: float = 0.06
    t_on_margin_hydro: float = 0.037
    t_on_margin_other: float = 0.013


@dataclass
class CostGasGenConfig:
    heat_rate: float
    co2_intensity: float


COST_LOW_EFF_GAS_GEN_CONFIG = CostGasGenConfig(heat_rate=11, co2_intensity=0.58)
COST_HIGH_EFF_GAS_GEN_CONFIG = CostGasGenConfig(heat_rate=6.5, co2_intensity=0.37)


def _normalize_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def daily_loan_payment(principle: float, rate: float, number_of_payments: int) -> float:
    return principle * (rate * (1 + rate) ** number_of_payments) / ((1 + rate) ** number_of_payments - 1)


CCGT_LOAN_PAYMENT = daily_loan_payment(CCGT_BUILD_COST, DISCOUNT_RATE, TIME)
NUC_LOAN_PAYMENT = daily_loan_payment(NUC_BUILD_COST, DISCOUNT_RATE, TIME)


def gas_price_path(config: GasPriceConfig, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = _normalize_rng(rng)
    prices = np.empty(TIME)
    prices[0] = config.starting_price
    for t in range(1, TIME):
        shock = rng.normal(0, config.std)
        prices[t] = max(0.0, prices[t - 1] + config.mean_reversion * (config.average_price - prices[t - 1]) + shock)
    return prices


def carbon_price_path(config: CarbonPriceConfig) -> np.ndarray:
    increasing = np.linspace(config.start, config.end, config.increase_days)
    flat = np.full(TIME - config.increase_days, config.end)
    return np.concatenate([increasing, flat])


def gas_gen_opex(gas_price: float, carbon_price: float, config: CostGasGenConfig) -> float:
    fuel_cost = config.heat_rate * gas_price
    carbon_cost = carbon_price * (config.co2_intensity - TIER_BENCHMARK)
    return fuel_cost + GAS_OPEX + carbon_cost


def electricity_price(
    gas_price: float,
    carbon_price: float,
    config: ElectricityPriceConfig,
    rng: Optional[np.random.Generator] = None,
) -> float:
    rng = _normalize_rng(rng)
    low_eff_gas_cost = gas_gen_opex(gas_price, carbon_price, COST_LOW_EFF_GAS_GEN_CONFIG)
    high_eff_gas_cost = gas_gen_opex(gas_price, carbon_price, COST_HIGH_EFF_GAS_GEN_CONFIG)
    hydro_cost = 80
    other_cost = 40

    base_price = (
        low_eff_gas_cost * config.t_on_margin_low_eff_gas
        + high_eff_gas_cost * config.t_on_margin_high_eff_gas
        + 0 * config.t_on_margin_zero_price
        + hydro_cost * config.t_on_margin_hydro
        + other_cost * config.t_on_margin_other
    )

    if rng.random() < config.scarcity_prob:
        return config.scarcity_price
    return base_price


def ccgt_cashflow(
    gas_price: float,
    electricity_price_value: float,
    carbon_price: float,
    discount: float,
) -> float:
    mwh_per_day = CCGT_CAPACITY * CCGT_CAPACITY_FACTOR * HOURS_PER_DAY
    revenue = electricity_price_value * mwh_per_day
    opex = gas_gen_opex(gas_price, carbon_price, COST_HIGH_EFF_GAS_GEN_CONFIG) * mwh_per_day
    return (revenue - opex - CCGT_FIXED_OM - CCGT_LOAN_PAYMENT) * discount


def nuc_cashflow(electricity_price_value: float, discount: float) -> float:
    mwh_per_day = NUC_CAPACITY * NUC_CAPACITY_FACTOR * HOURS_PER_DAY
    revenue = electricity_price_value * mwh_per_day
    opex = URANIUM_CONVERSION * URANIUM_PRICE * mwh_per_day + NUC_OPEX
    return (revenue - opex - NUC_FIXED_OM - NUC_LOAN_PAYMENT) * discount


def moving_average(values: np.ndarray, window: int = WINDOW) -> np.ndarray:
    if window <= 1:
        return values.copy()
    weights = np.ones(window)
    weighted_sum = np.convolve(values, weights, mode='same')
    sample_count = np.convolve(np.ones_like(values), weights, mode='same')
    return weighted_sum / sample_count


def simulate_single_scenario(
    gas_config: GasPriceConfig = GasPriceConfig(),
    carbon_config: CarbonPriceConfig = CarbonPriceConfig(),
    electricity_config: ElectricityPriceConfig = ElectricityPriceConfig(),
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray | float]:
    rng = _normalize_rng(rng)
    gas_prices = gas_price_path(gas_config, rng)
    carbon_prices = carbon_price_path(carbon_config)
    electricity_prices = np.empty(TIME)

    for t in range(TIME):
        electricity_prices[t] = electricity_price(gas_prices[t], carbon_prices[t], electricity_config, rng)

    gas_cash = np.empty(TIME)
    nuc_cash = np.empty(TIME)
    for t in range(TIME):
        gas_cash[t] = ccgt_cashflow(gas_prices[t], electricity_prices[t], carbon_prices[t], DISCOUNTS[t])
        nuc_cash[t] = nuc_cashflow(electricity_prices[t], DISCOUNTS[t])

    return {
        'gas_prices': gas_prices,
        'carbon_prices': carbon_prices,
        'electricity_prices': electricity_prices,
        'gas_cash': gas_cash,
        'nuc_cash': nuc_cash,
        'gas_prices_moving_avg': moving_average(gas_prices),
        'electricity_moving_avg': moving_average(electricity_prices),
        'gas_moving_avg': moving_average(gas_cash / DISCOUNTS),
        'nuc_moving_avg': moving_average(nuc_cash / DISCOUNTS),
        'ccgt_npv': float(np.sum(gas_cash)),
        'nuclear_npv': float(np.sum(nuc_cash)),
        'avg_gas_price': float(np.mean(gas_prices)),
        'avg_electricity_price': float(np.mean(electricity_prices)),
    }


def print_summary(metrics: Dict[str, np.ndarray | float]) -> None:
    print(f'CCGT NPV: {metrics["ccgt_npv"]:,.0f}')
    print(f'Nuclear NPV: {metrics["nuclear_npv"]:,.0f}')
    print(f'Avg Gas Price: {metrics["avg_gas_price"]:.4f}')
    print(f'Avg Electricity Price: {metrics["avg_electricity_price"]:.4f}')


def plot_scenario(metrics: Dict[str, np.ndarray | float]) -> None:
    fig = plt.figure(figsize=(5,5))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)
    time_years = np.arange(TIME) / DAYS_PER_YEAR

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_years, metrics['gas_moving_avg'], color='red', label='CCGT', linewidth=1)
    ax1.plot(time_years, metrics['nuc_moving_avg'], color='green', label='Nuclear', linewidth=1)
    ax1.set_ylabel('Daily Cash Flow (mil)')
    ax1.label_outer()
    ax1.legend(loc='upper right')

    ax3 = fig.add_subplot(gs[1], sharex=ax1)
    ax3.plot(time_years, metrics['electricity_moving_avg'], color='blue', linewidth=1)
    ax3.set_ylabel('Electricity ($/MWh)')
    ax3.label_outer()

    ax4 = fig.add_subplot(gs[2], sharex=ax1)
    ax4.plot(time_years, metrics['gas_prices_moving_avg'], color='black', linewidth=1)
    ax4.set_ylabel('Nat. Gas ($/GJ)')
    ax4.set_xlabel('Years')
    fig.align_ylabels([ax1, ax3, ax4])

    plt.savefig("fig2.png", dpi=1200, bbox_inches="tight")

    fig2 = plt.figure(figsize=(5,8))
    gs2 = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 1, 1, 1], hspace=0.1)

    cashflow_per_electricity = metrics['gas_moving_avg'] / metrics['electricity_moving_avg']
    nuc_cashflow_per_electricity = metrics['nuc_moving_avg'] / metrics['electricity_moving_avg']

    ax5 = fig2.add_subplot(gs2[0])
    ax5.plot(time_years, metrics['nuc_moving_avg'] / metrics['gas_prices_moving_avg'], color='green', label='Nuclear', linewidth=1)
    ax5.plot(time_years, metrics['gas_moving_avg'] / metrics['gas_prices_moving_avg'], color='red', label='CCGT', linewidth=1)
    ax5.set_ylabel('Cash Flow / Gas (GJ/day)')
    ax5.label_outer()
    ax5.legend(loc='upper right')

    ax6 = fig2.add_subplot(gs2[1])
    ax6.plot(time_years, nuc_cashflow_per_electricity, color='green', label='Nuclear', linewidth=1)
    ax6.plot(time_years, cashflow_per_electricity, color='red', label='CCGT', linewidth=1)
    ax6.set_ylabel('Cash Flow / Electricity (MWh/day)')
    ax6.label_outer()

    ax7 = fig2.add_subplot(gs2[2])
    ax7.plot(time_years, metrics['electricity_moving_avg'] / metrics['gas_prices_moving_avg'], color='purple', linewidth=1)
    ax7.set_ylabel('Electricity / Gas (MWh/GJ)')
    ax7.label_outer()

    ax8 = fig2.add_subplot(gs2[3])
    ax8.plot(time_years, metrics['electricity_moving_avg'], color='blue', linewidth=1)
    ax8.label_outer()

    ax9 = fig2.add_subplot(gs2[4])
    ax9.plot(time_years, metrics['gas_prices_moving_avg'], color='black', linewidth=1)
    ax9.set_xlabel('Years')
    fig2.align_ylabels([ax5, ax6, ax7, ax8, ax9])

    plt.savefig("fig3.png", dpi=1200, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a single electricity/gas economics scenario.')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting and print only the summary statistics.')
    args = parser.parse_args()

    metrics = simulate_single_scenario()
    print_summary(metrics)
    if not args.no_plot:
        plot_scenario(metrics)


if __name__ == '__main__':
    main()
