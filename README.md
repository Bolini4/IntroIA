# IntroIA
Introduction to IA / Bolin

## Installation.

1. Cloner le repository 
```bash 
git clone https://github.com/Bolini4/IntroIA.git
```
2. Construire le conteneur à partir du DockerFile

Le dossier Apprentissage_Python contient plusieurs actions possibles : 
- main.py -> Permettant d'entraîner un modèle à partir d'une base de données d'imgaes locales
- inference.py -> Permet de faire une inférence à partir du modèle crée précedemment.
- extract.py -> Permet d'extraire les poids et les biais du modèle sauvegardé (dans le dossier weightandbiases)
- infoDebug.py -> Donne des informations sur les couches pour debug la phase en C
- mainOptimisation.py -> Permet de faire de la visualisation pour trouver les meilleurs paramètres pour notre modèle.
- OLDMAIN.py -> Première version pour créer un modèle (obselete)