from argparse import ArgumentParser
from blocksparsednn.utils.file_utils import read_pickle_gz


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()

    print(read_pickle_gz(args.file))

