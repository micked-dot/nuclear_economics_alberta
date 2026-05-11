import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '16'

HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365
N_SIM = 1
GAS_OPEX = 5
TIER_BENCHMARK = 0.37
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
    average_price: float = 1.5
    std: float = 0.5
    mean_reversion: float = 0.1
    
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
    
COST_LOW_EFF_GAS_GEN_CONFIG = CostGasGenConfig(heat_rate=11, co2_intensity=0.58)
COST_HIGH_EFF_GAS_GEN_CONFIG = CostGasGenConfig(heat_rate=6.5, co2_intensity=0.37)

@dataclass
class CostNuclearGenConfig:
    fuel_conversion: float #kg/MWh

def gas_price_path(config: GasPriceConfig) -> np.ndarray:
    prices = [config.starting_price]
    for _ in range(TIME - 1):
        shock = np.random.normal(0, config.std)
        price = max(0.0, prices[-1] + config.mean_reversion * (config.average_price - prices[-1]) + shock)
        prices.append(price)
    return np.array(prices)

def carbon_price_path(config: CarbonPriceConfig) -> np.ndarray:
    increasing = np.linspace(config.start, config.end, config.increase_days)
    flat = np.full(TIME - config.increase_days, config.end)
    return np.concatenate([increasing, flat])

def gas_gen_opex(gas_price, carbon_price, config: CostGasGenConfig) -> float:
    fuel_cost = config.heat_rate * gas_price
    carbon_cost = carbon_price * (config.co2_intensity - TIER_BENCHMARK)
    return fuel_cost + GAS_OPEX + carbon_cost

def electricity_price(gas_price, carbon_price, config: ElectricityPriceConfig) -> float:
    low_eff_gas_cost = gas_gen_opex(gas_price, carbon_price, COST_LOW_EFF_GAS_GEN_CONFIG)
    high_eff_gas_cost = gas_gen_opex(gas_price, carbon_price, COST_HIGH_EFF_GAS_GEN_CONFIG)
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

def ccgt_cashflow(gas_price, electricity_price, carbon_price, discount) -> float:
    mwh_per_day = ccgt_capacity * ccgt_capacity_factor * HOURS_PER_DAY
    revenue =  electricity_price * mwh_per_day
    opex = gas_gen_opex(gas_price, carbon_price, COST_HIGH_EFF_GAS_GEN_CONFIG) * mwh_per_day
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



window = 180

gas_prices = gas_price_path(GasPriceConfig())
gas_prices_moving_avg = np.convolve(gas_prices, np.ones(window)/window, mode='same')

carbon_prices = carbon_price_path(CarbonPriceConfig())

electricity_prices = np.array([electricity_price(gas_prices[t], carbon_prices[t], ElectricityPriceConfig()) for t in range (TIME)])
electricity_moving_avg = np.convolve(electricity_prices, np.ones(window)/window, mode='same')

gas_cash = np.array([ccgt_cashflow(gas_prices[t], electricity_prices[t], carbon_prices[t], discounts[t]) for t in range (TIME)])
gas_cash_adjusted = gas_cash / discounts
gas_moving_avg = np.convolve(gas_cash_adjusted, np.ones(window)/window, mode='same')

nuc_cash = np.array([nuc_cashflow(electricity_prices[t], discounts[t]) for t in range (TIME)])
nuc_cash_adjusted = nuc_cash / discounts
nuc_moving_avg = np.convolve(nuc_cash_adjusted, np.ones(window)/window, mode='same')




print(f'CCGT NPV: {np.sum(gas_cash):,}')
print(f'Nuclear NPV: {np.sum(nuc_cash):,}')
print(f'Avg Gas Price: {np.sum(gas_prices)/TIME}')
print(f'Avg Electricity Price: {np.sum(electricity_prices)/TIME}')


fig = plt.figure()
gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)
time = np.linspace(0, TIME, TIME)

ax1 = fig.add_subplot(gs[0])
ax1.plot(time, gas_moving_avg, color='red', label='CCGT')
ax1.set_ylabel('Daily Cashflows')
ax1.label_outer()   
ax1.plot(time, nuc_moving_avg, color='green', label='Nuclear')
ax1.legend(loc='upper right')

ax3 = fig.add_subplot(gs[1], sharex=ax1)
ax3.plot(time, electricity_moving_avg, color='blue')
ax3.set_ylabel('Electricity')
ax3.label_outer()

ax4 = fig.add_subplot(gs[2], sharex=ax1)
ax4.plot(time, gas_prices_moving_avg, color='black')
ax4.set_ylabel('Natural Gas')
ax4.set_xlabel('Time (Days)')


fig2 = plt.figure()
gs = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 1, 1, 1], hspace=0.1)

ax5 = fig2.add_subplot(gs[0])
ax5.plot(time, nuc_moving_avg/gas_prices_moving_avg, color='green', label='Nuclear')
ax5.plot(gas_moving_avg/gas_prices_moving_avg, color='red', label='CCGT')
ax5.set_ylabel('Cashflow / Gas')
ax5.label_outer()
ax5.legend(loc='upper right')

ax6 = fig2.add_subplot(gs[1])
ax6.plot(time, nuc_moving_avg/electricity_moving_avg, color='green', label='Nuclear')
ax6.plot(gas_moving_avg/electricity_moving_avg, color='red', label='CCGT')
ax6.set_ylabel('Cashflow / Electricity')
ax6.label_outer()

ax7 = fig2.add_subplot(gs[2])
ax7.plot(time, electricity_moving_avg/gas_prices_moving_avg, color='purple')
ax7.set_ylabel('Electricity / Gas')
ax7.label_outer()

ax8 = fig2.add_subplot(gs[3])
ax8.plot(time, electricity_moving_avg, color='blue')
#ax8.set_ylabel('Electricity')
ax8.label_outer()

ax9 = fig2.add_subplot(gs[4])
ax9.plot(time, gas_prices_moving_avg, color='black')
#ax9.set_ylabel('Gas')

plt.show()