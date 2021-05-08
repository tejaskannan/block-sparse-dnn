Requires:
    - CUDA
    - tensorflow-gpu pip package or tensorflow installed from source

To build:
$ ./make-bsmm.sh

To include in python:
import tensorflow as tf
bsmm_module = tf.load_op_library("./bsmm.so")
bssm_module.BCSRMatMul(args...)

TROUBLESHOOTING:
Error msg: "could not find file {PYTHON PACKAGE PATH}/tensorflow/third-party/gpus/..."
    - softlink your cuda directory in tensorflow/third-party/gpus (make directory if necessary)