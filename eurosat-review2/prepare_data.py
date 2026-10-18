import argparse
from data import prepare_split

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Create a fixed 70/15/15 stratified split without copying images.')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', default='splits.json')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    prepare_split(args.data_dir, args.output, args.seed)
