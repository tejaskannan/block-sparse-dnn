import numpy as np
import re

NAME_REGEX = re.compile('[:/-]+')


def convert_name(name: str) -> str:
    """
    Converts a variable name to a C variable name.
    """
    name_part = name.split(':')[0]
    tokens = NAME_REGEX.split(name_part)
    return '_'.join(t.strip().upper() for t in tokens if len(t.strip()) > 0)


def convert_matrix_data(name: str, mat: np.ndarray, precision: int, width: int) -> str:
    """
    Converts a matrix to a constant C variable.

    Args:
        name: The name of the variable to write
        mat: The matrix data to write. Must be a 2d matrix.
        precision: The number of fractional bits for each value.
        width: The number of bits for each value.
    Returns:
        A representation of the data as a 1d C array.
    """
    assert len(mat.shape) == 2, 'Must provide a 2d array to convert'

    # Quantize the matrix values 
    fixed_point_mat = array_to_fixed_point(mat, precision=precision, width=width)

    # Write each array to a static C array representation
    c_arrays = [','.join(map(str, a)) for a in fixed_point_mat]
    c_matrix = '{{{0}}}'.format(','.join(c_arrays))

    # Create the variable name
    var_name = 'static int16_t {0}[{1}]'.format(name, np.prod(fixed_point_mat.shape))

    # Return the full variable declaration
    return '{0} = {1};'.format(var_name, c_matrix)


def convert_matrix(name: str, mat: np.ndarray, precision: int, width: int, is_msp: bool) -> str:
    """
    Converts the given matrix to a full C representation.

    Args:
        name: The name of the Tensorflow variable for this matrix
        mat: The name of the matrix to write.
        precision: The number of fractional bits
        width: The bit width of each fixed point value.
        is_msp: Whether to declare the variables for the MSP device (will
            place in FRAM).
    Returns:
        A few lines of C code creating the matrix.
    """
    components: List[str] = []
    
    # Create the matrix data name
    var_name = convert_name(name)
    data_name = '{0}_DATA'.format(var_name)

    if is_msp:
        persistent = '#pragma PERSISTENT({0})'.format(data_name)
        component.append(persistent)

    # Take transpose because we perform left-multiplication
    # in TF but right multiplication on the MSP
    mat = mat.T

    # Convert the data matrix
    mat_data = convert_matrix_data(data_name, mat=mat, precision=precision, width=width)
    components.append(mat_data)

    # Create the matrix variable
    num_rows, num_cols = mat.shape
    mat_var = 'static Matrix {0} = {{ {1}, {2}, {3} }};'.format(var_name, data_name, num_rows, num_cols)
    components.append(mat_var)

    return '\n'.join(components)


def array_to_fixed_point(arr: np.ndarray, precision: int, width: int) -> np.ndarray:
    """
    Converts the given (float) array to fixed point values.
    """
    multiplier = 1 << abs(precision)
    
    if precision > 0:
        quantized = arr * multiplier
    else:
        quantized = arr / multiplier

    quantized = np.round(quantized).astype(int)

    max_val = (1 << (width - 1)) - 1
    min_val = -max_val

    return np.clip(quantized, a_min=min_val, a_max=max_val).astype(int)
