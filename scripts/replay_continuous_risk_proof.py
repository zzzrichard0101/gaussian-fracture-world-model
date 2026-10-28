"""Replay a saved continuous-risk proof using only Python's standard library."""
import argparse
import json
from pathlib import Path

from wmfracture.continuous_risk import replay_proof


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('proof',type=Path)
    parser.add_argument('--expected-inputs',type=Path)
    parser.add_argument('--expected-action',type=float)
    args=parser.parse_args()
    result=json.loads(args.proof.read_text())
    expected=json.loads(args.expected_inputs.read_text()) if args.expected_inputs else None
    replay_proof(result,expected_inputs=expected,expected_action=args.expected_action)
    print(json.dumps({'proof_replayed':True,'decision':result['decision'],'lower_damage':result['lower_damage'],
        'upper_damage':result['upper_damage'],'scope':result['scope']},indent=2))


if __name__=='__main__':main()
