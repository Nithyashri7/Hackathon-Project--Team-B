# CLI entrypoint for the Kaggle Data Agent
import argparse
from agents.data_agent import DataAgent

def main():
    parser = argparse.ArgumentParser(description='Run Kaggle Data Agent')
    parser.add_argument('--topic', required=True, help='Search topic, e.g. "Heart Disease"')
    parser.add_argument('--download', action='store_true', help='Download the selected dataset')
    parser.add_argument('--limit', type=int, default=10, help='Number of candidate datasets to consider')
    parser.add_argument('--outdir', default='downloads', help='Output directory for downloaded datasets')
    args = parser.parse_args()

    agent = DataAgent()
    candidates = agent.search(topic=args.topic, limit=args.limit)
    print('Top candidates:')
    for i,c in enumerate(candidates,1):
        print(f'{i}. {c["title"]} — {c.get("ref", "")} — score={c.get("score")}')

    if args.download and candidates:
        chosen = candidates[0]
        path = agent.download_dataset(chosen, outdir=args.outdir)
        print('Downloaded to', path)

if __name__ == '__main__':
    main()
