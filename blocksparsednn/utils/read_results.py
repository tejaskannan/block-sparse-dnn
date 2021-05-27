from file_utils import read_jsonl_gz
import sys

print(list(read_jsonl_gz(sys.argv[1])))
