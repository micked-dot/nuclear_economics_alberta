import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '20'

HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365
N_SIM = 1
GAS_OPEX = 5
uranium_price = 0.94 #CAD/GJ
uranium_conversion = 10.8
TIME = 10*365 #days
ANNUAL_DISCOUNT_RATE = 0.05
DISCOUNT_RATE = (1 + ANNUAL_DISCOUNT_RATE) ** (1/DAYS_PER_YEAR) - 1

def daily_loan_payment(principle, rate, number_of_payments):
    return principle * (rate * (1+rate)**number_of_payments)/((1+rate)**number_of_payments-1)

ccgt_build_cost = 1.5 * 10 ** 9 #loan
ccgt_capacity_factor = 0.60 #% of time operating
ccgt_loan_payment = daily_loan_payment(ccgt_build_cost, DISCOUNT_RATE, TIME)
ccgt_capacity = 1000 #MW
ccgt_fixed_om = 30 * 1.37 * 10 ** 6 /365
ccgt_opex = 5 # $/MWh
    
nuc_build_cost = 8 * 10 ** 9#loan
nuc_loan_payment = daily_loan_payment(nuc_build_cost, DISCOUNT_RATE, TIME)
nuc_capacity_factor = 0.9 #% of time operating
nuc_capacity = 1000 #MW
nuc_fixed_om = 150 * 1.37 * 10 ** 6 / 365 # $/day
nuc_opex = 30 # $/MWh

@dataclass
class GasPriceConfig:
    starting_price: float = 1.5
    
@dataclass
class CarbonPriceConfig:
    increase_days: int = 4*365
    start: int = 95
    end: int = 170
    
@dataclass
class ElectricityPriceConfig:
    scarcity_prob: float =0.05
    scarcity_price: float = 500 #$/MWh
    t_on_margin_low_eff_gas: float = 0.68 #% of time on margin
    t_on_margin_high_eff_gas: float = 0.21 #% of time on margin
    t_on_margin_zero_price: float = 0.06 #% of time on margin
    t_on_margin_hydro: float = 0.037 #% of time on margin
    t_on_margin_other: float = 0.013 #% of time on margin
    
@dataclass
class CostGasGenConfig:
    heat_rate: float #GJ/MWh
    co2_intensity: float #tonnesCO2e/MWh
    tier_benchmark: float = 0.37
    
COST_LOW_EFF_GAS_GEN_CONFIG = CostGasGenConfig(heat_rate=11, co2_intensity=0.58)
COST_HIGH_EFF_GAS_GEN_CONFIG = CostGasGenConfig(heat_rate=6.5, co2_intensity=0.37)

@dataclass
class CostNuclearGenConfig:
    fuel_conversion: float #kg/MWh

def gas_price_path(config: GasPriceConfig) -> np.ndarray:
    return np.array([config.starting_price] * TIME)

def carbon_price_path(config: CarbonPriceConfig) -> np.ndarray:
    increasing = np.linspace(config.start, config.end, config.increase_days)
    flat = np.full(TIME - config.increase_days, config.end)
    return np.concatenate([increasing, flat])

def gas_gen_opex(gas_price, carbon_price, config: CostGasGenConfig) -> float:
    fuel_cost = config.heat_rate * gas_price
    carbon_cost = carbon_price * (config.co2_intensity - config.tier_benchmark)
    return fuel_cost + GAS_OPEX + carbon_cost

def electricity_price(gas_price, carbon_price, config: ElectricityPriceConfig, low_eff_config: CostGasGenConfig = None, high_eff_config: CostGasGenConfig = None) -> float:
    if low_eff_config is None:
        low_eff_config = COST_LOW_EFF_GAS_GEN_CONFIG
    if high_eff_config is None:
        high_eff_config = COST_HIGH_EFF_GAS_GEN_CONFIG
    low_eff_gas_cost = gas_gen_opex(gas_price, carbon_price, low_eff_config)
    high_eff_gas_cost = gas_gen_opex(gas_price, carbon_price, high_eff_config)
    zero_price_cost = 0
    hydro_cost = 80 #$/MWh - from FIGURE 18: Annual Achieved Price by Technology
    other_cost = 40 #$/MWh - from FIGURE 18: Annual Achieved Price by Technology
    
    base_price = (
        low_eff_gas_cost * config.t_on_margin_low_eff_gas +
        high_eff_gas_cost * config.t_on_margin_high_eff_gas +
        zero_price_cost * config.t_on_margin_zero_price +
        hydro_cost * config.t_on_margin_hydro +
        other_cost * config.t_on_margin_other)
    
    if np.random.rand() < config.scarcity_prob:
        return config.scarcity_price
    return base_price

def ccgt_cashflow(gas_price, electricity_price, carbon_price, discount, high_eff_config: CostGasGenConfig = None) -> float:
    if high_eff_config is None:
        high_eff_config = COST_HIGH_EFF_GAS_GEN_CONFIG
    mwh_per_day = ccgt_capacity * ccgt_capacity_factor * HOURS_PER_DAY
    revenue =  electricity_price * mwh_per_day
    opex = gas_gen_opex(gas_price, carbon_price, high_eff_config) * mwh_per_day
    fixed_cost = ccgt_fixed_om
    debt_payment = ccgt_loan_payment
    return (revenue - opex - fixed_cost - debt_payment) * discount

def nuc_cashflow(electricity_price, discount) -> float:
    mwh_per_day = nuc_capacity * nuc_capacity_factor * HOURS_PER_DAY
    revenue =  electricity_price * mwh_per_day
    opex = uranium_conversion * uranium_price * mwh_per_day + nuc_opex
    fixed_cost = nuc_fixed_om
    debt_payment = nuc_loan_payment
    return (revenue - opex - fixed_cost - debt_payment) * discount

discounts = np.array([1/((1+DISCOUNT_RATE)**t) for t in range(TIME)])

ccgt_npvs = []
nuc_npvs = []
avg_elec_prices = []
fixed_gas_prices = np.linspace(1, 10, 10)
tier_benchmarks = [0.37, 0]
ccgt_npvs_by_benchmark = []
nuc_npvs_by_benchmark = []
avg_elec_prices_by_benchmark = []

for benchmark in tier_benchmarks:
    ccgt_npvs_benchmark = []
    nuc_npvs_benchmark = []
    avg_elec_prices_benchmark = []
    
    for price in fixed_gas_prices:
        window = 180
        
        # Create gas configs with current benchmark
        low_eff_config = CostGasGenConfig(heat_rate=11, co2_intensity=0.58, tier_benchmark=benchmark)
        high_eff_config = CostGasGenConfig(heat_rate=6.5, co2_intensity=0.37, tier_benchmark=benchmark)

        gas_prices = gas_price_path(GasPriceConfig(starting_price=price))
        gas_prices_moving_avg = np.convolve(gas_prices, np.ones(window)/window, mode='same')

        carbon_prices = carbon_price_path(CarbonPriceConfig())

        electricity_prices = np.array([electricity_price(gas_prices[t], carbon_prices[t], ElectricityPriceConfig(), low_eff_config, high_eff_config) for t in range (TIME)])

        gas_cash = np.array([ccgt_cashflow(gas_prices[t], electricity_prices[t], carbon_prices[t], discounts[t], high_eff_config) for t in range (TIME)])

        nuc_cash = np.array([nuc_cashflow(electricity_prices[t], discounts[t]) for t in range (TIME)])

        ccgt_npvs_benchmark.append(np.sum(gas_cash))
        nuc_npvs_benchmark.append(np.sum(nuc_cash))
        avg_elec_prices_benchmark.append(np.sum(electricity_prices)/TIME)
    
    ccgt_npvs_by_benchmark.append(ccgt_npvs_benchmark)
    nuc_npvs_by_benchmark.append(nuc_npvs_benchmark)
    avg_elec_prices_by_benchmark.append(avg_elec_prices_benchmark)
    
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

for i, benchmark in enumerate(tier_benchmarks):
    alpha = 1.0 if i == 0 else 0.5
    ax1.plot(fixed_gas_prices, ccgt_npvs_by_benchmark[i], color='red', alpha=alpha, label=f'CCGT (benchmark={benchmark})')
    ax1.plot(fixed_gas_prices, nuc_npvs_by_benchmark[i], color='green', alpha=alpha, label=f'Nuclear (benchmark={benchmark})')
    ax2.plot(fixed_gas_prices, avg_elec_prices_by_benchmark[i], color='blue', alpha=alpha, label=f'Electricity Price (benchmark={benchmark})')

ax1.set_ylabel('NPV')
ax1.set_xlabel('Gas Price')
ax1.legend(loc='lower right')
ax1.label_outer()

ax2.set_ylabel('Electricity Price')
ax2.set_xlabel('Gas Price')

plt.show()