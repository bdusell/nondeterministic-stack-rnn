import argparse
import pathlib

from print_best import read_data_for_trial

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=pathlib.Path)
    args = parser.parse_args()

    trial = read_data_for_trial(args.input)
    valid_perplexity = trial.metrics['perplexity']
    print(f'{valid_perplexity:.2f}')

if __name__ == '__main__':
    main()
