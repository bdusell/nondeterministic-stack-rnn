import argparse
import pathlib

import numpy

from nsrnn.pytorch_tools.saver import read_logs

def read_epoch_times(path):
    prev_timestamp = None
    with read_logs(path) as events:
        for event in events:
            if event.type == 'start':
                prev_timestamp = event.timestamp
            elif event.type == 'epoch':
                curr_timestamp = event.timestamp
                time_diff = curr_timestamp - prev_timestamp
                prev_timestamp = curr_timestamp
                yield time_diff.total_seconds()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=pathlib.Path)
    args = parser.parse_args()

    mean_time = numpy.mean(list(read_epoch_times(args.input)))
    print(mean_time)

if __name__ == '__main__':
    main()
