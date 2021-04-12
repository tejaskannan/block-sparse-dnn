BIG_NUMBER = 1e7
SMALL_NUMBER = 1e-7

LOSS_OP = 'loss'
OPTIMIZER_OP = 'optimize'
PREDICTION_OP = 'prediction'
LOGITS_OP = 'logits'
DROPOUT_KEEP_RATE = 'dropout_keep_rate'
INPUTS = 'inputs'
OUTPUT = 'output'
TRAIN = 'train'
VAL = 'validation'
TEST = 'test'
FLOPS = 'flops'

TRAIN_BATCHES = 'train_batches'
INPUT_SHAPE = 'input_shape'
OUTPUT_SHAPE = 'output_shape'
SCALER = 'scaler'
DATASET_FOLDER = 'dataset_folder'
VOCAB = 'vocab'
SEQ_LENGTH = 'seq_length'
MAX_INPUT = 'max_input'

SPARSE_INDICES = 'sparse_indices'
SPARSE_DIMS = 'sparse_dims'
SPARSE_NAMES = 'sparse_names'
SPARSE_MASK = 'sparse_mask'

MODEL_FILE_FMT = 'model-{0}-model_best.pkl.gz'
HYPERS_FILE_FMT = 'model-hypers-{0}-model_best.pkl.gz'
METADATA_FILE_FMT = 'model-metadata-{0}-model_best.pkl.gz'
TRAIN_LOG_FMT = 'model-train-log-{0}-model_best.jsonl.gz'
TEST_LOG_FMT = 'model-test-log-{0}-model_best.jsonl.gz'
