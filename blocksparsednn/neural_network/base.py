import os.path
import tensorflow as tf
import numpy as np
import sklearn.metrics as metrics
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tensorflow.python.client import timeline
from typing import Dict, Any, Optional, List, Tuple, Set, Union

from blocksparsednn.dataset.dataset import Dataset, DataSeries, Batch
from blocksparsednn.utils.metrics import count_correct
from blocksparsednn.utils.tf_utils import get_optimizer
from blocksparsednn.utils.constants import OPTIMIZER_OP, LOSS_OP, PREDICTION_OP, INPUTS, OUTPUT, MAX_INPUT
from blocksparsednn.utils.constants import INPUT_SHAPE, OUTPUT_SHAPE, SCALER, LOGITS_OP, MODEL_FILE_FMT, HYPERS_FILE_FMT
from blocksparsednn.utils.constants import METADATA_FILE_FMT, TRAIN_LOG_FMT, DATASET_FOLDER, TRAIN_BATCHES, BIG_NUMBER
from blocksparsednn.utils.file_utils import save_pickle_gz, read_pickle_gz, make_dir, save_jsonl_gz, extract_model_name


DEFAULT_HYPERS = {
    'batch_size': 16,
    'learning_rate': 0.001,
    'learning_rate_decay': 0.99,
    'optimizer': 'adam',
    'gradient_clip': 1,
    'num_epochs': 50,
    'patience': 50,
    'should_normalize_inputs': True
}


class NeuralNetwork:

    def __init__(self, name: str, hypers: Dict[str, Any], log_device: bool = False):
        self._name = name
        self._sess = tf.Session(graph=tf.Graph(),
                                config=tf.ConfigProto(log_device_placement=log_device))
        self._ops: Dict[str, tf.Tensor] = dict()
        self._placeholders: Dict[str, tf.placeholder] = dict()
        self._metadata: Dict[str, Any] = dict()
        self._is_made = False
        self._is_frozen = False

        # Set the random seed to get reproducible results
        with self._sess.graph.as_default():
            tf.set_random_seed(8547)

        # Overwrite default hyper-parameters
        self._hypers = {key: val for key, val in DEFAULT_HYPERS.items()}
        self._hypers.update(**hypers)

    @property
    def name(self) -> str:
        return self._name

    @property
    def learning_rate(self) -> float:
        return float(self._hypers['learning_rate'])

    @property
    def learning_rate_decay(self) -> float:
        return float(self._hypers['learning_rate_decay'])

    @property
    def batch_size(self) -> int:
        return int(self._hypers['batch_size'])

    @property
    def gradient_clip(self) -> float:
        return float(self._hypers['gradient_clip'])

    @property
    def num_epochs(self) -> int:
        return int(self._hypers['num_epochs'])

    @property
    def patience(self) -> int:
        return int(self._hypers['patience'])

    @property
    def optimizer_name(self) -> str:
        return str(self._hypers['optimizer'])

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self._metadata['input_shape']

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return self._metadata['output_shape']

    def get_trainable_vars(self) -> List[tf.Variable]:
        return list(self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    def count_parameters(self) -> int:
        return sum((np.prod(var.get_shape()) for var in self.get_trainable_vars()))

    def make_placeholders(self, is_frozen: bool):
        """
        Creates the placeholders for this model.
        """
        raise NotImplementedError()

    def make_graph(self, is_train: bool, is_frozen: bool):
        """
        Creates the computational graph.
        """
        raise NotImplementedError()

    def make_loss(self):
        """
        Creates the loss function. This should give a value
        to self._ops[LOSS_OP].
        """
        raise NotImplementedError()

    def post_epoch_step(self, epoch: int, has_ended: bool):
        """
        Executes a post epoch step to evolve the model

        Args:
            epoch: The index of the current epoch
            has_ended: Whether training has terminated
        """
        pass

    def batch_to_feed_dict(self, batch: Batch, is_train: bool) -> Dict[tf.placeholder, np.ndarray]:
        raise NotImplementedError() 

    def count_flops(self) -> int:
        """
        Counts the number of floating point operations in one forward pass of the model.
        """
        assert self._is_frozen, 'Must freeze the model to count floating point ops'
        run_metadata = tf.RunMetadata()

        with self._sess.graph.as_default():
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.profiler.profile(self._sess.graph, run_meta=run_metadata, cmd='op', options=opts)

        return flops.total_float_ops

    def load_metadata(self, dataset: Dataset):
        """
        Loads metadata.
        """
        # Build the input normalization object
        scaler = StandardScaler()

        batch_generator = dataset.minibatch_generator(series=DataSeries.TRAIN,
                                                      batch_size=self.batch_size,
                                                      should_shuffle=False)

        batch_count = 0
        max_input = -BIG_NUMBER
        class_set: Set[int] = set()
        for batch in batch_generator:
            batch_samples = len(batch.inputs)

            # Fit the normalization object on the batch. We reshape to a 2d matrix
            # for compatibility reasons
            scaler = scaler.partial_fit(batch.inputs.reshape(batch_samples, -1))

            # Get the maximum input. This is really only useful for vocab-based inputs
            # for which we create embedding vectors.
            max_input = max(np.max(batch.inputs), max_input)

            input_shape = batch.inputs.shape[1:]
            class_set.update(batch.output.astype(int))
            batch_count += 1

        self._metadata[INPUT_SHAPE] = input_shape
        self._metadata[OUTPUT_SHAPE] = len(class_set)  # Output shape is the number of classes
        self._metadata[SCALER] = scaler
        self._metadata[MAX_INPUT] = max_input
        self._metadata[DATASET_FOLDER] = dataset.dataset_folder
        self._metadata[TRAIN_BATCHES] = batch_count

    def make(self, is_train: bool, is_frozen: bool):
        """
        Builds the model.
        """
        if self._is_made:
            return

        self._is_frozen = is_frozen

        with self._sess.graph.as_default():
            self.make_placeholders(is_frozen=is_frozen)

            self.make_graph(is_train=is_train, is_frozen=is_frozen)
            assert PREDICTION_OP in self._ops, 'Must create a prediction operation'

            if not is_frozen:
                self.make_loss()
                assert LOSS_OP in self._ops, 'Must create a loss operation'

                self._global_step = tf.Variable(0, dtype=tf.int64, trainable=False)  # Tracks the number of training steps

                self._optimizer = get_optimizer(name=self.optimizer_name,
                                                learning_rate=self.learning_rate,
                                                global_step=self._global_step,
                                                decay_rate=self.learning_rate_decay,
                                                decay_steps=self._metadata[TRAIN_BATCHES])
                self.make_training_step()
                assert OPTIMIZER_OP in self._ops, 'Must create an optimizer operation'

        self._is_made = True

    def make_training_step(self):
        """
        Creates the training step for gradient descent
        """
        trainable_vars = self.get_trainable_vars()

        # Compute the gradients
        gradients = tf.gradients(self._ops[LOSS_OP], trainable_vars)

        # Clip Gradients
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip)

        # Prune None values from the set of gradients and apply gradient weights
        pruned_gradients = [(grad, var) for grad, var in zip(clipped_gradients, trainable_vars) if grad is not None]

        # Apply clipped gradients
        optimizer_op = self._optimizer.apply_gradients(pruned_gradients)

        # Increment the global step (handles learning rate decay)
        increment_op = tf.assign_add(self._global_step, 1)

        # Add optimization and increment steps to the operations
        self._ops[OPTIMIZER_OP] = tf.group(optimizer_op, increment_op)

    def init(self):
        """
        Initializes the trainable variables.
        """
        with self._sess.graph.as_default():
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    def execute(self, ops: Union[List[str], str], feed_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Executes the given operations.

        Args:
            ops: Names of operations to execute
            feed_dict: Dictionary supplying values to placeholders
        Returns:
            A dictionary of op_name -> op_result
        """
        assert self._is_made, 'Must call make() first'
        assert not self._is_frozen, 'Cannot run a frozen graph'

        # Turn operations into a list
        ops_list = ops
        if not isinstance(ops, list):
            ops_list = [ops]

        with self._sess.graph.as_default():
            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()

            ops_to_run = {op_name: self._ops[op_name] for op_name in ops_list}
            # results = self._sess.run(ops_to_run, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            results = self._sess.run(ops_to_run, feed_dict=feed_dict)

            #run_timeline = timeline.Timeline(run_metadata.step_stats)
            #chrome_trace = run_timeline.generate_chrome_trace_format()
            #with open('run_meta_timeline.json', 'w') as fout:
            #    fout.write(chrome_trace)

        return results

    def test(self, dataset: Dataset, batch_size: Optional[int]) -> Dict[str, float]:
        """
        Executes the model on the test fold of the given dataset.
        """
        # Execute the model on the testing samples
        test_batch_size = batch_size if batch_size is not None else self.batch_size
        test_generator = dataset.minibatch_generator(series=DataSeries.TEST,
                                                     batch_size=test_batch_size,
                                                     should_shuffle=False)

        pred_list: List[int] = []  # Record predictions
        label_list: List[int] = []  # Record labels
        loss_sum = 0.0
        num_samples = 0

        start_time = datetime.now()
        test_exec_time = 0.0
        exec_batches = 0.0

        test_ops = [PREDICTION_OP, LOSS_OP]

        for batch_idx, batch in enumerate(test_generator):
            feed_dict = self.batch_to_feed_dict(batch, is_train=False)

            batch_start = time.perf_counter()
            batch_result = self.execute(ops=test_ops, feed_dict=feed_dict)
            batch_end = time.perf_counter()

            if batch_idx > 0:
                test_exec_time += (batch_end - batch_start)
                exec_batches += 1

            predicted_probs = batch_result[PREDICTION_OP]  # [B, K]
            batch_pred = np.argmax(predicted_probs, axis=-1).astype(int)  # [B]

            pred_list.extend(batch_pred)
            label_list.extend(batch.output.reshape(-1).astype(int))

            loss_sum += batch_result[LOSS_OP] * len(batch.output)
            num_samples += len(batch.output)

        end_time = datetime.now()

        preds = np.array(pred_list)  # [M]
        labels = np.array(label_list)  # [M]

        print('Exec Batches: {0}, Total Time: {1}'.format(exec_batches, test_exec_time))

        # Compute the testing metrics
        accuracy = metrics.accuracy_score(y_true=labels, y_pred=preds)
        f1_score = metrics.f1_score(y_true=labels, y_pred=preds, average='macro')
        precision = metrics.precision_score(y_true=labels, y_pred=preds, average='macro')
        recall = metrics.recall_score(y_true=labels, y_pred=preds, average='macro')
        loss = loss_sum / num_samples
        time_per_batch = test_exec_time / max(exec_batches, 1.0)

        return {
            'accuracy': accuracy,
            'loss': loss,
            'macro_f1_score': f1_score,
            'macro_precision': precision,
            'macro_recall': recall,
            'duration': str(end_time - start_time),
            'start_time': start_time.strftime('%Y-%m-%d-%H-%M-%S'),
            'end_time': end_time.strftime('%Y-%m-%d-%H-%M-%S'),
            'time_per_batch': time_per_batch
        }

    def train(self, dataset: Dataset, save_folder: str, should_print: bool, should_save_model: bool) -> str:
        """
        Trains the neural network on the given dataset.
        """
        # Load the meta-data from the train data
        self.load_metadata(dataset)

        # Build the model
        self.make(is_train=True, is_frozen=False)

        # Initialize the variables
        self.init()

        # Create a name for this training run based on the current date and time
        start_time = datetime.now()
        model_name = '{0}-{1}-{2}'.format(self.name, dataset.name, start_time.strftime('%Y-%m-%d-%H-%M-%S'))

        # Create lists to track the training and validation metrics
        train_loss: List[float] = []
        train_accuracy: List[float] = []
        val_loss: List[float] = []
        val_accuracy: List[float] = []

        # Track best model and early stopping
        best_val_accuracy = 0
        num_not_improved = 0

        # Get the warm-up epochs
        warmup = self._hypers.get('warmup', 0)
        assert warmup >= 0, 'Must have a non-negative warmup'

        num_params = self.count_parameters()
        print('Training model with {0} parameters.'.format(num_params))

        # Augment the save directory with the data-set name and model name
        # for clarity
        make_dir(save_folder)

        save_folder = os.path.join(save_folder, dataset.name)
        make_dir(save_folder)

        save_folder = os.path.join(save_folder, self.name)
        make_dir(save_folder)

        train_start = time.time()
        start_time = datetime.now()

        for epoch in range(self.num_epochs):

            print('==========')
            print('Epoch: {0}/{1}'.format(epoch + 1, self.num_epochs))
            print('==========')

            # Execute the training operations
            train_generator = dataset.minibatch_generator(series=DataSeries.TRAIN,
                                                          batch_size=self.batch_size,
                                                          should_shuffle=True)
            epoch_train_loss = 0.0
            epoch_train_correct = 0
            num_train_samples = 0

            train_ops = [LOSS_OP, LOGITS_OP, OPTIMIZER_OP]
            for batch_idx, train_batch in enumerate(train_generator):
                feed_dict = self.batch_to_feed_dict(train_batch, is_train=True)

                train_batch_results = self.execute(feed_dict=feed_dict, ops=train_ops)

                batch_samples = len(train_batch.inputs)

                # Add to the running loss. The returned loss is the average over the batch
                # so we multiply by the batch size to get the total loss.
                epoch_train_loss += train_batch_results[LOSS_OP] * batch_samples
                epoch_train_correct += count_correct(logits=train_batch_results[LOGITS_OP],
                                                     labels=train_batch.output)
                num_train_samples += batch_samples

                if should_print:
                    train_loss_so_far = epoch_train_loss / num_train_samples
                    train_acc_so_far = epoch_train_correct / num_train_samples
                    print('Train Batch {0}: Loss -> {1:.5f}, Accuracy -> {2:.5f}'.format(batch_idx + 1, train_loss_so_far, train_acc_so_far), end='\r')

            # Clear the line after the epoch
            if should_print:
                print()

            # Log the train loss
            train_loss.append(epoch_train_loss / num_train_samples)
            train_accuracy.append(epoch_train_correct / num_train_samples)

            # Execute the validation operations
            val_generator = dataset.minibatch_generator(series=DataSeries.VAL,
                                                        batch_size=self.batch_size,
                                                        should_shuffle=False)
            epoch_val_loss = 0.0
            epoch_val_correct = 0
            num_val_samples = 0

            val_ops = [LOSS_OP, LOGITS_OP]
            for batch_idx, val_batch in enumerate(val_generator):
                feed_dict = self.batch_to_feed_dict(val_batch, is_train=False)
                val_batch_results = self.execute(feed_dict=feed_dict, ops=val_ops)

                batch_samples = len(val_batch.inputs)

                epoch_val_loss += val_batch_results[LOSS_OP] * batch_samples
                epoch_val_correct += count_correct(logits=val_batch_results[LOGITS_OP],
                                                   labels=val_batch.output)
                num_val_samples += batch_samples

                if should_print:
                    val_loss_so_far = epoch_val_loss / num_val_samples
                    val_acc_so_far = epoch_val_correct / num_val_samples
                    print('Validation Batch {0}: Loss -> {1:.5f}, Accuracy -> {2:.5f}'.format(batch_idx + 1, val_loss_so_far, val_acc_so_far), end='\r')

            if should_print:
                print()

            # Log the validation results
            epoch_val_accuracy = epoch_val_correct / num_val_samples
            val_loss.append(epoch_val_loss / num_val_samples)
            val_accuracy.append(epoch_val_accuracy)

            # Check if we see improved validation performance
            should_save = False
            is_in_warmup = epoch < warmup

            if not is_in_warmup:
                if epoch_val_accuracy > best_val_accuracy:
                    should_save = True
                    num_not_improved = 0
                    best_val_accuracy = epoch_val_accuracy
                else:
                    num_not_improved += 1

            has_ended = bool(epoch == self.num_epochs - 1)
            if num_not_improved >= self.patience:
                if should_print:
                    print('Exiting due to early stopping.')

                has_ended = True

            # Save model if specified
            if should_save and should_save_model:
                if should_print:
                    print('Saving...')
                self.save(save_folder=save_folder, model_name=model_name)

            # Execute the post epoch step that is present in some models. We do this
            # after the model saving because the post epoch step may add random, untrained connections.
            # we don't want these to be present during testing.
            self.post_epoch_step(epoch=epoch, has_ended=has_ended)

            if has_ended:
                break

        # Log the training results
        end_time = datetime.now()
        train_end = time.time()

        train_time = train_end - train_start

        train_log = {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'duration': str(end_time - start_time),  # Total time (accounting for validation and saving)
            'start_time': start_time.strftime('%Y-%m-%d-%H-%M-%S'),
            'end_time': end_time.strftime('%Y-%m-%d-%H-%M-%S'),
            'train_time': train_time  # Actual training execution time in seconds
        }

        train_log_path = os.path.join(save_folder, TRAIN_LOG_FMT.format(model_name))
        save_jsonl_gz([train_log], train_log_path)

        return model_name

    def save(self, save_folder: str, model_name: str):
        """
        Saves the model parameters and associated training parameters
        in the given folder.
        """
        make_dir(save_folder)

        # Save the hyper-parameters
        hypers_file = os.path.join(save_folder, HYPERS_FILE_FMT.format(model_name))
        save_pickle_gz(self._hypers, hypers_file)

        # Save the meta-data
        metadata_file = os.path.join(save_folder, METADATA_FILE_FMT.format(model_name))
        save_pickle_gz(self._metadata, metadata_file)

        # Save the trainable parameters
        with self._sess.graph.as_default():
            trainable_vars = self._sess.run({var.name: var for var in self.get_trainable_vars()})
            
            model_file = os.path.join(save_folder, MODEL_FILE_FMT.format(model_name))
            save_pickle_gz(trainable_vars, model_file)

    @classmethod
    def restore(cls, model_file: str, is_frozen: bool):
        save_folder, model_file_name = os.path.split(model_file)
        model_name = extract_model_name(model_file_name)

        # Fetch the hyper-parameters
        hypers_path = os.path.join(save_folder, HYPERS_FILE_FMT.format(model_name))
        hypers = read_pickle_gz(hypers_path)

        # Intialize the new model
        name = model_name.split('-')[0]
        network = cls(name=name, hypers=hypers)

        # Fetch the meta-data
        metadata_path = os.path.join(save_folder, METADATA_FILE_FMT.format(model_name))
        metadata = read_pickle_gz(metadata_path)
        network._metadata = metadata

        # Build the model
        network.make(is_train=False, is_frozen=is_frozen)

        # Initialize the model
        network.init()

        # Restore the trainable variables
        with network._sess.graph.as_default():
            model_file = os.path.join(save_folder, MODEL_FILE_FMT.format(model_name))
            saved_vars = read_pickle_gz(model_file)

            trainable_vars = network.get_trainable_vars()

            assign_ops = []
            for var in trainable_vars:
                var_name = var.name

                if var_name not in saved_vars:
                    print('WARNING: No value for {0}'.format(var_name))
                else:
                    assign_ops.append(tf.assign(var, saved_vars[var_name]))

            network._sess.run(assign_ops)

        return network
