#include "main.h"

#define INPUT_BUFFER_LEN 100
#define LABEL_BUFFER_LEN 20


int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Must provide input files.\n");
        return 1;
    }

    // Create input string buffers
    char inputBuffer[INPUT_BUFFER_LEN];
    char labelBuffer[LABEL_BUFFER_LEN];

    // Create an input vector buffer
    int16_t inputData[NUM_INPUT_FEATURES * VECTOR_COLS];
    Matrix inputs = { inputData, NUM_INPUT_FEATURES, VECTOR_COLS };

    FILE *inputFile = fopen(argv[1], "r");
    FILE *labelFile = fopen(argv[2], "r");

    uint16_t lineIdx = 0;
    char *featureToken;
    int16_t feature, label, pred;
    uint16_t featureIdx;

    int numSamples = 0;
    int numCorrect = 0;

    while (fgets(inputBuffer, INPUT_BUFFER_LEN, inputFile)) {
        fgets(labelBuffer, LABEL_BUFFER_LEN, labelFile);

        // Parse the input vector
        featureToken = strtok(inputBuffer, ",");
        featureIdx = 0;

        while (featureToken) {
            feature = (int16_t) atoi(featureToken);
            inputData[VECTOR_INDEX(featureIdx)] = feature;

            featureIdx += 1;
            featureToken = strtok(NULL, ",");
        }

        // Parse the label
        label = (int16_t) atoi(labelBuffer);

        // Compute the prediction
        #ifdef IS_BLOCK_SPARSE
        pred = block_diagonal_mlp(&inputs, FIXED_POINT_PRECISION);
        #elif defined(IS_SPARSE)
        pred = sparse_mlp(&inputs, FIXED_POINT_PRECISION);
        #elif defined(IS_DENSE)
        pred = dense_mlp(&inputs, FIXED_POINT_PRECISION);
        #endif

        numCorrect += (int16_t) (label == pred);
        numSamples += 1;
    }

    printf("Accuracy: %d / %d\n", numCorrect, numSamples);

    return 0;
}
