import math
from typing import List


def get_num_nonzero(in_units: int, out_units: int, sparsity: float) -> int:
    """
    Computes the number of nonzero values for the given layer. A connection
    has nonzero value with prob sparsity * (in + out)/(in * out). This value
    is then multiplied by the total number of possible connections, (in * out),
    to get the nonzero values. Thus, the total number of nonzero connections
    is int(sparsity * (in + out)).
    """
    units_sum = in_units + out_units
    num_nonzero = int(round(sparsity * units_sum))

    return max(min(num_nonzero, in_units * out_units), 1)
