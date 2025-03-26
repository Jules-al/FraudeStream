# Détection de Fraude Bancaire

Cette application permet de détecter les fraudes bancaires en utilisant des modèles d'apprentissage automatique.

## Fonctionnalités

- Analyse exploratoire des données
- Prétraitement des données
- Entraînement de plusieurs modèles de ML
- Interface utilisateur avec Streamlit
- Visualisation des résultats et métriques

## Installation
``` 
1. Cloner le repository
2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

Pour lancer l'application :
```bash
streamlit run app/main.py
```

## Structure du Projet

```
.
├── app/
│   ├── main.py
│   ├── models/
│   │   └── model_training.py
│   ├── preprocessing/
│   │   └── data_processor.py
│   └── utils/
│       └── helpers.py
├── data/
│   └── README.md
├── models/
│   └── README.md
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
└── README.md
``` # fraudev
