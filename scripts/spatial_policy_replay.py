"""Replay decisions from public readouts/catalog only; no evaluation labels."""
import argparse
import json
from pathlib import Path

from wmfracture.spatial_observation import PhysicalCatalog, choose_public
from spatial_observation_evaluate import POLICY_KEYS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='docs/results/spatial-observation-config.json')
    parser.add_argument('--catalog', default='docs/results/spatial-observation-catalog.json')
    parser.add_argument('--decisions', default='docs/results/spatial-observation-decisions.json')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    parameters = json.loads(Path(args.catalog).read_text())
    decisions = json.loads(Path(args.decisions).read_text())
    bank = PhysicalCatalog(parameters, settings['actions_m'], settings['future_drives_m'], settings['cycle_increments'])
    ps = {key:settings[key] for key in POLICY_KEYS}
    selected = decisions if args.limit is None else decisions[:args.limit]
    for row in selected:
        action, posterior, plan = choose_public(row['policy'], bank, row['public_actions'], row['public_observed_m'],
            row['positions_m'], row['sigma_m'], row['turn'], ps)
        assert action == row['action'], row['trajectory_id']
        assert int(posterior['feasible'].sum()) == row['feasible_count'], row['trajectory_id']
        assert plan['proposal'] == row['plan']['proposal'], row['trajectory_id']
    print(f'Replayed {len(selected)} spatial decisions from public observations/catalog only; no truth/evaluation file read')


if __name__ == '__main__':
    main()
