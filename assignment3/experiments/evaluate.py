"""Checkpoint evaluation entry point reserved for Loc and the team."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.parse_args()
    parser.exit(2, "Evaluation is not implemented. No metrics or predictions were produced.\n")


if __name__ == "__main__":
    main()
