import numpy as np
import re
import os.path
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, List, DefaultDict, Any

from blocksparsednn.utils.file_utils import read_pickle_gz, extract_model_name
from blocksparsednn.utils.constants import METADATA_FILE_FMT, HYPERS_FILE_FMT, INPUT_SHAPE
from blocksparsednn.conversion.convert_utils import convert_matrix, convert_name


BLOCK_REGEX = re.compile('kernel-([0-9]+):0')
WIDTH = 16
PRECISION = 10


def group_block_kernels(var_names: List[str]) -> DefaultDict[str, Dict[int, str]]:
    """
    Groups matrix sub-blocks corresponding to the same variable.
    """
    groups: DefaultDict[str, Dict[int, str]] = defaultdict(dict)

    for var_name in var_names:
        name_tokens = var_name.split('/')
        group_name, rest = name_tokens[0], name_tokens[1]

        # Extract the block index number
        block_match = BLOCK_REGEX.match(rest)
        if not block_match:
            continue

        block_idx = int(block_match.group(1))

        groups[group_name][block_idx] = var_name

    return groups


def convert_block_layer(group_name: str,
                        group_vars: Dict[int, str],
                        weights: Dict[str, np.ndarray],
                        rows: List[int],
                        cols: List[int],
                        input_size: int,
                        output_size: int,
                        block_size: int,
                        is_msp: bool) -> str:
    """
    Converts the block matrix into a C representation.
    """
    components: List[str] = []

    block_names: List[str] = []
    for block_idx in range(len(group_vars)):
        block_var = group_vars[block_idx]

        block_mat = convert_matrix(name=block_var,
                                   mat=weights[block_var],
                                   precision=PRECISION,
                                   width=WIDTH,
                                   is_msp=is_msp)

        components.append(block_mat)
        block_names.append(convert_name(block_var))

    var_name = convert_name(group_name)
    
    # Create the array of blocks
    block_var_name = '{0}_BLOCKS'.format(var_name)

    block_addresses = ['&{0}'.format(n) for n in block_names]
    block_array = 'static Matrix *{0}[] = {{{1}}};'.format(block_var_name, ','.join(block_addresses))
    components.append(block_array)

    # Write the row and column arrays. We switch the order
    # because we do operations in the MSP using the transpose matrix.
    row_name = '{0}_ROWS'.format(var_name)
    row_array = '{{{0}}}'.format(','.join(map(lambda t: str(t * block_size), cols)))
    row_var = 'static uint16_t {0}[] = {1};'.format(row_name, row_array)
    components.append(row_var)

    col_name = '{0}_COLS'.format(var_name)
    col_array = '{{{0}}}'.format(','.join(map(lambda t: str(t * block_size), rows)))
    col_var = 'static uint16_t {0}[] = {1};'.format(col_name, col_array)
    components.append(col_var)

    # Create the block sparse matrix
    mat_name = '{0}_KERNEL'.format(var_name)
    mat_var = 'static BlockSparseMatrix {0} = {{ {1}, {2}, {3}, {4}, {5}, {6} }};'.format(mat_name, block_var_name, len(rows), output_size, input_size, row_name, col_name)
    components.append(mat_var)

    # Create the bias vector
    bias_name = '{0}/bias:0'.format(group_name)
    bias_var = convert_matrix(name=bias_name,
                              mat=weights[bias_name],
                              precision=PRECISION,
                              width=WIDTH,
                              is_msp=is_msp)
    components.append(bias_var)

    return '\n'.join(components)


def convert_block_sparse_network(weights: Dict[str, np.ndarray],
                                 metadata: Dict[str, Any],
                                 hypers: Dict[str, Any],
                                 is_msp: bool):
    """
    Convert a block sparse matrix a C header file for the C model implementation.
    """
    # List to hold all variables
    components: List[str] = []


    # Extract the block sparse layers
    block_groups = group_block_kernels(list(weights.keys()))

    # Start the input units at the input shape
    input_size = metadata[INPUT_SHAPE][-1]

    for i, (group_name, group_vars) in enumerate(block_groups.items()):

        # Get the output size
        output_size = hypers['hidden_units'][i]

        block_declaration = convert_block_layer(group_name=group_name,
                                                group_vars=group_vars,
                                                weights=weights,
                                                input_size=input_size,
                                                output_size=output_size,
                                                rows=metadata['block-rows'][group_name],
                                                cols=metadata['block-cols'][group_name],
                                                block_size=hypers['block_size'],
                                                is_msp=is_msp)

        components.append(block_declaration)

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


def write_result(variables: str, output_file: str, is_msp: bool):

    with open(output_file, 'w') as fout:

        # Write the imports
        fout.write('#include <stdint.h>\n')

        if is_msp:
            fout.write('#include <msp430.h>\n')

        fout.write('#include "math/matrix.h"\n')

        # Write the header guard
        fout.write('#ifndef NEURAL_NETWORK_PARAMS_H\n')
        fout.write('#define NEURAL_NETWORK_PARAMS_H\n')

        # Write the network type and other constants
        fout.write('#define IS_BLOCK_SPARSE\n')
        fout.write('#define FIXED_POINT_PRECISION {0}\n'.format(PRECISION))

        # Write the variables
        fout.write(variables)
        fout.write('\n')

        fout.write('#endif')



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-file', type=str, required=True)
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

    declaration = convert_block_sparse_network(weights=weights,
                                 metadata=metadata,
                                 hypers=hypers,
                                 is_msp=False)

    
    write_result(declaration, 'neural_network_parameters.h', is_msp=False)
