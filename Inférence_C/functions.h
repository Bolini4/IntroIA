#include <stddef.h>
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#define ROWS 28
#define COLS 28
#define FLATTENED_SIZE (ROWS * COLS)

#define OUTPUT_SIZE 1092  // Définir la taille de sortie
#define INPUT_SIZE 728  // Remplacer VOTRE_TAILLE_D_ENTREE par la taille de votre entrée


typedef struct {
    int inputSize;
    int outputSize;
    float **weights;
    float *biases;
} DenseLayer;

void loadWeightsAndBiases(DenseLayer *layer, const char *weightsFile, const char *biasesFile, int inputSize, int outputSize);
void flattenImage(unsigned char **image, int flattenedImage[FLATTENED_SIZE]);
float *CalculerLayer1092(DenseLayer *layer, int input[]);
float *CalculerLayer10(DenseLayer *layer, float input[]);
float *CalculerLayer10Final(DenseLayer *layer, float input[]);
void softmax(float *input, size_t input_len);
float relu(float x);

#endif /* FLATTEN_IMAGE_H */
