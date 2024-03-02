#include <stdio.h>
#include <stdlib.h>

#include "Bmp2Matrix.h"
#include "functions.h"

int main(int argc, char* argv[]){

    DenseLayer Layer1, Layer2, Layer3;

    printf("Loading weights and biases...\n");
    loadWeightsAndBiases(&Layer1, "./weightandbiases/layer_1_weights.txt", "./weightandbiases/layer_1_biases.txt", 784, 64);
    loadWeightsAndBiases(&Layer2, "./weightandbiases/layer_2_weights.txt", "./weightandbiases/layer_2_biases.txt", 64, 1092);
    loadWeightsAndBiases(&Layer3, "./weightandbiases/layer_3_weights.txt", "./weightandbiases/layer_3_biases.txt", 1092, 10);
    printf("Weights and biases loaded successfully\n");




   BMP bitmap;
   FILE* pFichier=NULL;

   pFichier=fopen("0_1.bmp", "rb");     //Ouverture du fichier contenant l'image
   if (pFichier==NULL) {
       printf("%s\n", "0_1.bmp");
       printf("Erreur dans la lecture du fichier\n");
   }
   LireBitmap(pFichier, &bitmap);
   fclose(pFichier);               //Fermeture du fichier contenant l'image

   ConvertRGB2Gray(&bitmap);

   int flatImage[FLATTENED_SIZE];
    flattenImage(bitmap.mPixelsGray, flatImage);

printf("Flattened image: \n");

//print layer1 weights
for (int i = 0; i < 64; i++) {
    for (int j = 0; j < 784; j++) {
        printf("%f\n", Layer1.weights[i][j]);
    }
}

    float *output1 = CalculerLayer1092(&Layer1, flatImage);
    printf("Output of layer 1: \n");
    // float *output2 = CalculerLayer10(&Layer2, output1);
    // float *output3 = CalculerLayer10Final(&Layer3, output2);

    printf("Output of layer 2: \n");
for (int i = 0; i < 10; i++) {
    printf("%f\n", output1[i]);
}

    DesallouerBMP(&bitmap);

   return 0;
}