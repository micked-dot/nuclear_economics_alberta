import argparse
from contextlib import contextmanager
from typing import Dict, Iterable

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import original


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '10'

FIXED_GAS_PRICES = np.linspace(1, 10, 10)
TIER_BENCHMARKS = [0.37, 0]


@contextmanager
def tier_benchmark(value: float):
    previous = original.TIER_BENCHMARK
    original.TIER_BENCHMARK = value
    try:
        yield
    finally:
        original.TIER_BENCHMARK = previous


def simulate_fixed_gas_price(gas_price: float, benchmark: float, n_sim: int, seed: int) -> Dict[str, float]:
    results = []
    gas_config = original.GasPriceConfig(
        starting_price=gas_price,
        average_price=gas_price,
        std=0.0,
    )

    with tier_benchmark(benchmark):
        for sim in range(n_sim):
            rng = np.random.default_rng(seed + sim)
            results.append(original.simulate_single_scenario(gas_config=gas_config, rng=rng))

    return {
        'ccgt_npv': float(np.mean([result['ccgt_npv'] for result in results])),
        'nuclear_npv': float(np.mean([result['nuclear_npv'] for result in results])),
        'avg_electricity_price': float(np.mean([result['avg_electricity_price'] for result in results])),
    }


def run_gas_price_sweep(
    gas_prices: Iterable[float] = FIXED_GAS_PRICES,
    benchmarks: Iterable[float] = TIER_BENCHMARKS,
    n_sim: int = 1,
    seed: int = 42,
) -> Dict[float, Dict[str, list[float]]]:
    sweep_results = {}

    for benchmark in benchmarks:
        benchmark_results = {
            'ccgt_npvs': [],
            'nuclear_npvs': [],
            'avg_electricity_prices': [],
        }

        for gas_price in gas_prices:
            metrics = simulate_fixed_gas_price(gas_price, benchmark, n_sim, seed)
            benchmark_results['ccgt_npvs'].append(metrics['ccgt_npv'])
            benchmark_results['nuclear_npvs'].append(metrics['nuclear_npv'])
            benchmark_results['avg_electricity_prices'].append(metrics['avg_electricity_price'])

        sweep_results[benchmark] = benchmark_results

    return sweep_results


def plot_gas_price_sweep(
    gas_prices: Iterable[float],
    sweep_results: Dict[float, Dict[str, list[float]]],
    output_file: str,
) -> None:
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    styles = {
        0.37: {'alpha': 1.0, 'red': 'red', 'green': 'green', 'blue': 'blue'},
        0: {'alpha': 0.6, 'red': 'red', 'green': 'green', 'blue': 'blue'},
    }

    for benchmark, results in sweep_results.items():
        style = styles.get(benchmark, {'alpha': 1.0, 'red': 'red', 'green': 'green', 'blue': 'blue'})
        ax1.plot(
            gas_prices,
            np.array(results['ccgt_npvs']) / 1e9,
            color=style['red'],
            alpha=style['alpha'],
            label='CCGT' if benchmark == 0.37 else f'CCGT (No Tier)',
        )
        ax1.plot(
            gas_prices,
            np.array(results['nuclear_npvs']) / 1e9,
            color=style['green'],
            alpha=style['alpha'],
            label='Nuclear' if benchmark == 0.37 else 'Nuclear (No Tier)',
        )
        ax2.plot(
            gas_prices,
            results['avg_electricity_prices'],
            color=style['blue'],
            alpha=style['alpha'],
            label='Electricity Price' if benchmark == 0.37 else 'Electricity Price (No Tier)',
        )

    ax1.set_ylabel('NPV (bil)')
    ax1.legend(loc='lower right')
    ax1.label_outer()

    ax2.set_ylabel('Electricity Price ($/MWh)')
    ax2.set_xlabel('Gas Price ($/GJ)')
    ax2.legend(loc='lower right')
    ax2.label_outer()

    fig.align_ylabels([ax1, ax2])
    plt.savefig(output_file, dpi=1200, bbox_inches='tight')


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot CCGT and nuclear NPVs across fixed gas prices.')
    parser.add_argument('--n-sim', type=int, default=1, help='Number of simulations to average at each gas price.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible simulations.')
    parser.add_argument('--output', default='gas_price_sweep.png', help='Figure output path.')
    parser.add_argument('--no-show', action='store_true', help='Save the figure without opening the plot window.')
    args = parser.parse_args()

    sweep_results = run_gas_price_sweep(n_sim=args.n_sim, seed=args.seed)
    plot_gas_price_sweep(FIXED_GAS_PRICES, sweep_results, args.output)

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
