# Prédiction des préférences humaines entre réponses de LLM

Projet de machine learning multiclasses visant à prédire si un évaluateur humain préférera la réponse A, la réponse B, ou considérera les deux réponses comme équivalentes.

## Résultat

- Log Loss public : **1,00880**
- Classement au moment de la capture : **45e sur 248 équipes actives**
- Position approximative : **Top 18,1 %**
- Notebook Kaggle final : **Version 17**
- Résultat enregistré en **juillet 2026**

> Cette compétition utilise un classement glissant sur deux mois. Le rang indiqué correspond aux équipes actives au moment des captures d’écran et peut évoluer à mesure que de nouvelles soumissions apparaissent ou que d’anciennes expirent.

### Résultat de la soumission

![Soumission réussie](images/submission-score.png)

### Position dans le classement

![Position dans le classement](images/leaderboard-rank-45.png)

### Taille du classement actif

![Nombre d’équipes actives](images/leaderboard-248.png)

## Problème

L’objectif était de prédire la préférence humaine entre deux réponses générées par des modèles de langage pour un même prompt.

Le modèle produit trois probabilités :

- la réponse A gagne ;
- la réponse B gagne ;
- égalité.

La métrique de la compétition est la Log Loss multiclasses. La qualité de la calibration des probabilités est donc plus importante que le simple choix de la classe la plus probable.

## Modèle final

Le système final combine :

- trois classifieurs LightGBM entraînés avec différentes graines aléatoires ;
- une régression logistique multinomiale ;
- une augmentation des données par inversion A/B ;
- une augmentation au moment de l’inférence par inversion A/B ;
- une calibration par température.

Pondération finale :

- ensemble LightGBM : **94 %**
- régression logistique : **6 %**
- température : **0,99**

## Ingénierie des variables

Le pipeline a généré plus de **700 variables candidates** et en a retenu **143** pour le modèle final.

Les principales familles de variables comprennent :

- longueur et structure des textes ;
- similarité lexicale entre le prompt et les réponses ;
- similarité entre les réponses A et B ;
- embeddings sémantiques multilingues ;
- répétition et lisibilité ;
- respect des consignes et du format demandé ;
- couverture des sous-questions ;
- indicateurs de raisonnement et de factualité ;
- indicateurs liés au code et à la qualité technique ;
- ton, sécurité, refus et respect du rôle demandé.

Les variables sémantiques ont été générées avec :

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

## Validation

- Log Loss en validation croisée : environ **1,0092**
- Log Loss public Kaggle : **1,00880**

La proximité entre le score de validation et le score public suggère que la procédure de validation s’est correctement généralisée.

## Structure du dépôt

```text
.
├── notebooks/
│   └── final-inference-v17.ipynb
├── images/
│   ├── leaderboard-rank-45.png
│   ├── leaderboard-248.png
│   └── submission-score.png
├── reports/
├── README.md
├── requirements.txt
└── .gitignore

```

## Reproductibilité

Le dépôt ne contient pas :

- les données de la compétition Kaggle ;
- les modèles entraînés au format `.joblib` ;
- les checkpoints volumineux du modèle d’embeddings ;
- les fichiers intermédiaires au format `.parquet`.

Ces fichiers sont exclus pour des raisons de licence, de taille et de reproductibilité.

Pour reproduire le pipeline, il faut récupérer les données depuis Kaggle et fournir localement les artefacts de modèles nécessaires.

## Limites

- Le modèle tabulaire ne vérifie pas directement la véracité factuelle des réponses.
- Les variables liées au code analysent sa structure mais ne l’exécutent pas.
- De nombreuses variables reposent sur des règles linguistiques conçues manuellement.
- L’encodeur sémantique reste plus léger qu’un grand transformer entraîné directement sur cette tâche.
- Le classement Kaggle étant glissant sur deux mois, le rang affiché peut évoluer.

## Points clés

Ce projet met en œuvre un pipeline complet de machine learning comprenant :

- ingénierie de variables ;
- validation croisée ;
- sélection de variables ;
- combinaison de plusieurs modèles ;
- calibration des probabilités ;
- gestion de la symétrie par inversion A/B ;
- inférence hors ligne sur un test caché ;
- déploiement et débogage sur Kaggle.