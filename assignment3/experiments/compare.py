"""Experiment comparison entry point reserved for Loc and the team."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="results/runs")
    parser.parse_args()
    parser.exit(2, "Comparison is not implemented. No experiment results have been generated.\n")


if __name__ == "__main__":
    main()
