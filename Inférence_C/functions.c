#include "functions.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stddef.h>

void flattenImage(unsigned char **image, int flattenedImage[FLATTENED_SIZE]) {
    int index = 0;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            flattenedImage[index++] = (int)image[i][j];
        }
    }
}


float relu(float x) {
    return x > 0 ? x : 0;
}



float *CalculerLayer1092(DenseLayer *layer, int input[]) {
    // Allouer dynamiquement de la mémoire pour le vecteur de sortie
    int outputSize = 1092;
    float *output = malloc(outputSize * sizeof(float));
    if (output == NULL) {
        exit(1);
    }

    for (int i = 0; i < outputSize; i++) {
        output[i] = layer->biases[i];
        for (int j = 0; j < layer->inputSize; j++) {
            //print layer->inputSize
            printf("%d\n", layer->inputSize);
            output[i] += layer->weights[i][j] * input[j];
            //ça passe 64 fois ici avant de crash ...
        }
        output[i] = relu(output[i]);
    }

    return output;
}


float *CalculerLayer10(DenseLayer *layer, float input[]) {
    // Allouer dynamiquement de la mémoire pour le vecteur de sortie
    int outputSize = 10;
    float *output = malloc(outputSize * sizeof(float));
    if (output == NULL) {
        exit(1);
    }

    for (int i = 0; i < outputSize; i++) {
        output[i] = layer->biases[i];
        for (int j = 0; j < layer->inputSize; j++) {
            output[i] += layer->weights[i][j] * input[j];
        }
        output[i] = relu(output[i]);
    }

    return output;
}

float *CalculerLayer10Final(DenseLayer *layer, float input[]) {
    // Allouer dynamiquement de la mémoire pour le vecteur de sortie
    int outputSize = 10;
    float *output = malloc(outputSize * sizeof(float));
    if (output == NULL) {
        exit(1);
    }

    // Calcul de la somme pondérée
    float sum = 0.0;
    for (int i = 0; i < outputSize; i++) {
        output[i] = layer->biases[i];
        for (int j = 0; j < layer->inputSize; j++) {
            output[i] += layer->weights[i][j] * input[j];
        }
        sum += expf(output[i]);
    }

    // Application de la fonction softmax
    softmax(output, outputSize);

    return output;
}




void loadWeightsAndBiases(DenseLayer *layer, const char *weightsFile, const char *biasesFile, int inputSize, int outputSize) {
    // Charger les poids à partir du fichier weightsFile
    layer->weights = (float**)malloc(outputSize * sizeof(float*));
    for (int i = 0; i < outputSize; i++) {
        layer->weights[i] = (float*)malloc(inputSize * sizeof(float));
    }

    layer->biases = (float*)malloc(outputSize * sizeof(float));

    printf("Loading weights from %s\n", weightsFile);
    FILE *weights_fp = fopen(weightsFile, "r");
    if (weights_fp == NULL) {
        fprintf(stderr, "Erreur lors de l'ouverture du fichier %s\n", weightsFile);
        exit(1);
    }
    
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            if (fscanf(weights_fp, "%f ", &layer->weights[i][j]) != 1) {
                fprintf(stderr, "Erreur lors de la lecture des poids\n");
                exit(1);
            }
        }
    }
    
    fclose(weights_fp);

    // Charger les biais à partir du fichier biasesFile
    FILE *biases_fp = fopen(biasesFile, "r");
    if (biases_fp == NULL) {
        fprintf(stderr, "Erreur lors de l'ouverture du fichier %s\n", biasesFile);
        exit(1);
    }
    for (int i = 0; i < outputSize; i++) {
        fscanf(biases_fp, "%f", &layer->biases[i]);
    }
    fclose(biases_fp);
}

void softmax(float *input, size_t input_len) {
  assert(input);

  float m = -INFINITY;
  for (size_t i = 0; i < input_len; i++) {
    if (input[i] > m) {
      m = input[i];
    }
  }

  float sum = 0.0;
  for (size_t i = 0; i < input_len; i++) {
    sum += expf(input[i] - m);
  }

  float offset = m + logf(sum);
  for (size_t i = 0; i < input_len; i++) {
    input[i] = expf(input[i] - offset);
  }
}