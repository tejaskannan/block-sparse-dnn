import csv
import matplotlib.pyplot as plt
from argparse import ArgumentParser


THRESHOLD = 1e6


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--trace-file', type=str, required=True)
    parser.add_argument('--num-operations', type=float, required=True)
    args = parser.parse_args()

    currents = []

    # Get the time (in nanoseconds) in which the operations took place
    with open(args.trace_file, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')

        start_time = None
        end_time = None

        for idx, line in enumerate(reader):
            if idx > 0:
                current = float(line[1])

                currents.append(current)

                if (current > THRESHOLD) and (start_time is None):
                    start_time = float(line[0]) / 1e9  # Time in seconds
                elif (current < THRESHOLD) and (start_time is not None):
                    end_time = float(line[0]) / 1e9  # Time in seconds
                    break

    time_delta = end_time - start_time

    print('Seconds: {0}'.format(time_delta))
    print('Sec / Op: {0}'.format(time_delta / args.num_operations))
    print('Ops / Sec: {0}'.format(args.num_operations / time_delta))

    plt.plot(list(range(len(currents))), currents)
    plt.show()
