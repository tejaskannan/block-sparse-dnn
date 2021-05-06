CCBIN=/usr/bin/gcc
CC=$(CCBIN) -Wall -Wextra -std=c99 -pedantic -g -Wno-unused-variable -fsanitize=address

default: model

model: utils/utils.h utils/neural_network_utils.h math/matrix.h neural_network_parameters.h math/fixed_point_ops.h math/matrix_ops.h layers/layers.h neural_network.h main.h math/fixed_point_ops.c math/matrix_ops.c layers/layers.c neural_network.c main.c
	$(CC) -o model math/fixed_point_ops.c math/matrix_ops.c layers/layers.c neural_network.c main.c

matrix_test: utils/utils.h math/fixed_point_ops.h math/matrix.h math/matrix_ops.h tests/matrix_tests.h math/fixed_point_ops.c math/matrix_ops.c tests/matrix_tests.c
	$(CC) -o matrix_test math/fixed_point_ops.c math/matrix_ops.c tests/matrix_tests.c

fixed_point_test: utils/utils.h tests/fixed_point_tests.h tests/fixed_point_tests.c math/fixed_point_ops.h math/fixed_point_ops.c
	$(CC) -o fixed_point_tests math/fixed_point_ops.c tests/fixed_point_tests.c

clean: model
	rm -f *.o model fixed_point_tests matrix_test *.dSYM
