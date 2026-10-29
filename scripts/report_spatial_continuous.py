"""Derived scientific figure only; original estimates and decisions unchanged."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]/'artifacts/spatial-continuous'
    predictions = json.loads((root/'predictions.json').read_text())
    report = json.loads((root/'report.json').read_text())
    config = report['config']; powers = config['nested_powers']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for sigma in (.000001, .00002):
        valid, ess = [], []
        for power in powers:
            selected = [r for r in predictions if r['sigma_m'] == sigma and r['power'] == power]
            valid.append(sum(r['available'] for r in selected)/len(selected))
            ess.append(float(np.median([r['ess'] for r in selected])))
        label = f'sigma={sigma*1000:g} mm'
        axes[0].plot([2**p for p in powers], valid, 'o-', label=label)
        axes[1].plot([2**p for p in powers], ess, 'o-', label=label)
    axes[0].set(ylabel='Fraction with nonempty sampled likelihood', ylim=(-.03, 1.03))
    axes[1].set(ylabel='Median effective sample size (not an error bound)')
    for axis in axes:
        axis.set_xscale('log', base=2); axis.set_xlabel('Sobol nodes per scramble'); axis.legend(fontsize=8)
    fig.suptitle('Uniform continuous prior: 32 fixed histories; no case passes all convergence screens')
    fig.tight_layout(); fig.savefig(root/'comparison.png', dpi=170); plt.close(fig)


if __name__ == '__main__': main()
