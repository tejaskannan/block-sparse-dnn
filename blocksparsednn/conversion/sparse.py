import numpy as np
import re
import os.path
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, List, DefaultDict, Any

from blocksparsednn.utils.file_utils import read_pickle_gz, extract_model_name
from blocksparsednn.utils.constants import METADATA_FILE_FMT, HYPERS_FILE_FMT, INPUT_SHAPE
from blocksparsednn.conversion.convert_utils import convert_matrix, convert_name, array_to_fixed_point


BLOCK_REGEX = re.compile('kernel-([0-9]+):0')
WIDTH = 16
PRECISION = 9


def convert_sparse_layer(layer_name: str,
                         weights: Dict[str, np.ndarray],
                         coordinates: np.ndarray,
                         input_size: int,
                         output_size: int,
                         is_msp: bool) -> str:
    """
    Converts the COO sparse matrix into a C representation.
    """
    components: List[str] = []
    var_name = convert_name(layer_name)

    # Create the 1d weight array
    weight_name = '{0}_DATA'.format(var_name)
    if is_msp:
        persistent = '#pragma PERSISTENT({0})'.format(weight_name)
        components.append(persistent)

    kernel_name = '{0}/kernel:0'.format(layer_name)

    fp_weights = array_to_fixed_point(weights[kernel_name],
                                      precision=PRECISION,
                                      width=WIDTH)
    weight_array = '{{{0}}}'.format(','.join(map(str, fp_weights)))
    weight_var = 'static int16_t {0}[] = {1};'.format(weight_name, weight_array)
    components.append(weight_var)

    # Create the row and column arrays
    row_name = '{0}_ROWS'.format(var_name)
    row_array = '{{{0}}}'.format(','.join(map(str, coordinates[:, 0])))
    row_var = 'static uint16_t {0}[] = {1};'.format(row_name, row_array)
    components.append(row_var)

    col_name = '{0}_COLS'.format(var_name)
    col_array = '{{{0}}}'.format(','.join(map(str, coordinates[:, 1])))
    col_var = 'static uint16_t {0}[] = {1};'.format(col_name, col_array)
    components.append(col_var)

    # Create the block sparse matrix
    mat_name = '{0}_KERNEL'.format(var_name)
    nnz = len(coordinates)
    mat_var = 'static SparseMatrix {0} = {{ {1}, {2}, {3}, {4}, {5}, {6} }};'.format(mat_name, weight_name, output_size, input_size, row_name, col_name, nnz)
    components.append(mat_var)

    # Create the bias vector
    bias_name = '{0}/bias:0'.format(layer_name)
    bias_var = convert_matrix(name=bias_name,
                              mat=weights[bias_name],
                              precision=PRECISION,
                              width=WIDTH,
                              is_msp=is_msp)
    components.append(bias_var)

    return '\n'.join(components)


def convert_sparse_network(weights: Dict[str, np.ndarray],
                           metadata: Dict[str, Any],
                           hypers: Dict[str, Any],
                           is_msp: bool):
    """
    Convert a block sparse matrix a C header file for the C model implementation.
    """
    # List to hold all variables
    components: List[str] = []

    # Start the input units at the input shape
    input_size = metadata[INPUT_SHAPE][-1]

    for i, output_size in enumerate(hypers['hidden_units']):

        # Convert the sparse variable
        layer_name = 'hidden_{0}'.format(i)

        sparse_layer = convert_sparse_layer(layer_name=layer_name,
                                            weights=weights,
                                            input_size=input_size,
                                            output_size=output_size,
                                            coordinates=metadata['sparse_indices'][layer_name],
                                            is_msp=is_msp)

        components.append(sparse_layer)

        # Reset the input size as we progress
        input_size = output_size

    # Include the output layer
    output_kernel_name = 'output/kernel:0'
    output_kernel = convert_matrix(name=output_kernel_name,
                                   mat=weights[output_kernel_name],
                                   precision=PRECISION,
                                   width=WIDTH,
                                   is_msp=is_msp)
    components.append(output_kernel)

    output_bias_name = 'output/bias:0'
    output_bias = convert_matrix(name=output_bias_name,
                                 mat=weights[output_bias_name],
                                 precision=PRECISION,
                                 width=WIDTH,
                                 is_msp=is_msp)
    components.append(output_bias)

    return '\n'.join(components)


def write_result(variables: str, num_input_features: int, output_file: str, is_msp: bool):

    with open(output_file, 'w') as fout:

        # Write the imports
        fout.write('#include <stdint.h>\n')
        fout.write('#include "math/matrix.h"\n')

        # Write the header guard
        fout.write('#ifndef NEURAL_NETWORK_PARAMS_H\n')
        fout.write('#define NEURAL_NETWORK_PARAMS_H\n')

        # Write the network type and other constants
        fout.write('#define IS_SPARSE\n')
        fout.write('#define FIXED_POINT_PRECISION {0}\n'.format(PRECISION))
        fout.write('#define NUM_INPUT_FEATURES {0}\n'.format(num_input_features))

        if is_msp:
            fout.write('#define IS_MSP\n')

        # Write the variables
        fout.write(variables)
        fout.write('\n')

        fout.write('#endif')



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-file', type=str, required=True)
    parser.add_argument('--is-msp', action='store_true')
    args = parser.parse_args()

    # Read the weights
    weights = read_pickle_gz(args.model_file)

    # Get the meta-data
    model_name = extract_model_name(args.model_file)
    save_folder = os.path.split(args.model_file)[0]
    metadata_path = os.path.join(save_folder, METADATA_FILE_FMT.format(model_name))
    metadata = read_pickle_gz(metadata_path)

    # Get the hyper parameters
    hypers_path = os.path.join(save_folder, HYPERS_FILE_FMT.format(model_name))
    hypers = read_pickle_gz(hypers_path)

    declaration = convert_sparse_network(weights=weights,
                                         metadata=metadata,
                                         hypers=hypers,
                                         is_msp=args.is_msp)
    
    write_result(declaration,
                 output_file='neural_network_parameters.h',
                 num_input_features=metadata['input_shape'][-1],
                 is_msp=args.is_msp)
