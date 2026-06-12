# PFE_PMSM_Diagnostic
PFE PMSM - Diagnostic Prédictif des Systèmes de Propulsion Électrique

Auteur : Layla Essadiq

École : ENSAM Rabat - Filière Énergie Électrique et Industrie Numérique

Entreprise : AFD Technologies (Accenture) - Projet Stellantis

Année universitaire : 2025/2026

📋 Description du Projet

Ce projet développe une méthodologie complète de diagnostic prédictif pour moteurs PMSM (Permanent Magnet Synchronous Motor) exploités dans les véhicules électriques du groupe Stellantis. L'approche repose exclusivement sur les signaux électriques déjà embarqués (courants de phase, vitesse, couple) sans capteur additionnel.

Architecture du pipeline :

plain

Signaux électriques → Prétraitement → 17 indicateurs (FFT/STFT/Wavelet) 
→ Health Index (0-1) → SVM (classification) → Modèle prédictif (horizon 10 pas)
→ Rapport de diagnostic automatique

🚀 Démarrage Rapide 
Prérequis
Python 3.12.2 (version exacte testée)
Git
~500 Mo d'espace disque
Installation
bash


# 1. Cloner le dépôt
git clone https://github.com/ton-username/PFE_PMSM_Diagnostic.git
cd PFE_PMSM_Diagnostic



# 2. Créer l'environnement virtuel (recommandé)
python3.12 -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate



# 3. Installer les dépendances exactes
pip install -r requirements.txt

Vérification de l'installation

bash

python -c "import numpy; print('NumPy:', numpy.__version__)"

python -c "import scipy; print('SciPy:', scipy.__version__)"

python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"

📁 Structure du Projet

plain

PFE_PMSM/

│

├── README.md                    ← Ce fichier

├── requirements.txt             ← Versions exactes des dépendances

├── LICENCE.md                 ← Licence MIT (voir ci-dessous)

│

├── 📂 scripts/                ← Scripts Python principaux (Annexe C)

│   ├── pmsm_simulation.py           # Chapitre III - Simulation PMSM

│   ├── pmsm_analyse_frequentielle.py # Chapitre IV - Analyse spectrale

│   ├── pmsm_health_index.py         # Chapitre V - Health Index

│   ├── bacha.py                     # Chapitre VI - Validation données réelles

│   ├── explorer_dataset.py          # Exploration dataset Bacha

│   ├── pmsm_perspective1.py         # Comparaison sim vs réel

│   ├── pmsm_perspective2_ml.py      # Chapitre VII - Machine Learning

│   └── pmsm_interface.py            # Interface de visualisation
│
├── 📂 dataset_reel/           ← Données expérimentales (Annexe D)
│   └── converted_dataset.csv        # Dataset Bacha et al. (2024)
│                                      # 10,892 mesures réelles
│                                      # Source: DOI 10.5281/zenodo.13974503
│
├── 📂 resultats/              ← Sorties générées automatiquement
│   ├── ch3_*.png                    # Figures simulation
│   ├── ch4_*.png / ch4_*.csv        # Features & heatmaps
│   ├── ch5_*.png / ch5_*.txt        # Health Index & rapports
│   ├── ch6_*.png / ch6_*.csv        # Validation réelle
│   └── perspective2_*.png           # ML & prédictions
│
└── 📂 docs/                   ← Documentation complémentaire
    ├── rapport_PFE_Essadiq.pdf      # Rapport complet
    └── presentation_soutenance.pptx  # Slides soutenance
🔄 Ordre d'Exécution Complet
Exécution automatique (recommandé)
bash
# Exécuter tout le pipeline en une commande
python run_pipeline.py
Exécution manuelle étape par étape
Table
Étape	Script	Commande	Temps	Output
1	Simulation PMSM	python scripts/pmsm_simulation.py	~2 min	resultats/ch3_*.png
2	Analyse fréquentielle	python scripts/pmsm_analyse_frequentielle.py	~3 min	resultats/ch4_*.csv/png
3	Health Index	python scripts/pmsm_health_index.py	~1 min	resultats/ch5_*.png/txt
4	Exploration dataset	python scripts/explorer_dataset.py	~30 sec	Aperçu terminal
5	Validation réelle	python scripts/bacha.py	~2 min	resultats/ch6_*.png
6	Comparaison	python scripts/pmsm_perspective1.py	~1 min	resultats/ch6_comparaison_sim_reel.png
7	Machine Learning	python scripts/pmsm_perspective2_ml.py	~5 min	resultats/perspective2_*.png
8	Interface	python scripts/pmsm_interface.py	~30 sec	Dashboard interactif
Exemple de commande complète (Windows PowerShell)
powershell
# Depuis la racine du projet
$env:PYTHONPATH = "."
python scripts/pmsm_simulation.py
python scripts/pmsm_analyse_frequentielle.py
python scripts/pmsm_health_index.py
python scripts/bacha.py
python scripts/pmsm_perspective2_ml.py
Exemple de commande complète (Linux/macOS)
bash
# Depuis la racine du projet
export PYTHONPATH=.
python scripts/pmsm_simulation.py && \
python scripts/pmsm_analyse_frequentielle.py && \
python scripts/pmsm_health_index.py && \
python scripts/bacha.py && \
python scripts/pmsm_perspective2_ml.py
🎯 Outputs Attendus
Après exécution complète, vous devez obtenir exactement ces fichiers :
Figures Chapitre III (Simulation)
resultats/ch3_simulation_nominale.png — 6 sous-graphes régime nominal
resultats/ch3_comparaison_short_circuit.png — Court-circuit vs sain
resultats/ch3_comparaison_phase_imbalance.png — Déséquilibre vs sain
resultats/ch3_comparaison_magnet_demagnetization.png — Démagnétisation vs sain
resultats/ch3_comparaison_bearing_fault.png — Roulement vs sain
resultats/ch3_comparaison_wiring_fault.png — Câblage vs sain
Figures Chapitre IV (Analyse fréquentielle)
resultats/ch4_fft_comparaison.png — Spectres FFT comparatifs
resultats/ch4_stft_nominal.png — Spectrogramme nominal
resultats/ch4_stft_bearing_fault.png — Spectrogramme roulement
resultats/ch4_wavelet_nominal.png — Décomposition wavelet sain
resultats/ch4_wavelet_short_circuit.png — Décomposition wavelet défaut
resultats/ch4_feature_heatmap.png — Heatmap 17 indicateurs
resultats/ch4_feature_dataset.csv — Dataset features (6 lignes × 17 colonnes)
Figures Chapitre V (Health Index)
resultats/ch5_health_index_barres.png — HI par scénario
resultats/ch5_poids_indicateurs.png — Poids variance inter-scénarios
resultats/ch5_degradation_temporelle.png — Courbes dégradation progressive
resultats/ch5_dashboard.png — Tableau de bord 6 jauges
resultats/ch5_rapport_diagnostic.txt — Rapport textuel automatique
Figures Chapitre VI (Validation réelle)
resultats/ch6_signaux_reel.png — 9 signaux réels Bacha
resultats/ch6_heatmap_reel.png — Heatmap données réelles
resultats/ch6_hi_reel_barres.png — HI moyen par condition réelle
resultats/ch6_hi_reel_evolution.png — Évolution HI fenêtre par fenêtre
resultats/ch6_comparaison_sim_reel.png — Comparaison côte à côte
Figures Chapitre VII (Machine Learning)
resultats/perspective2_svm_confusion.png — Matrice confusion SVM
resultats/perspective2_validation_croisee.png — 5-folds CV
resultats/perspective2_svm_pca.png — Frontières décision PCA 2D
resultats/perspective2_importance_features.png — Importance par permutation
resultats/perspective2_lstm_prediction.png — Prédiction HI 10 pas
resultats/perspective2_rapport_ml.txt — Rapport ML
🔒 Reproductibilité Garantie
Seeds fixes (random_state)
Tous les scripts utilisent random_state=42 pour garantir la reproductibilité :
Table
Opération	Script	Ligne	Seed
Augmentation données	pmsm_perspective2_ml.py	Ligne 85	random_state=42
Split train/test	pmsm_perspective2_ml.py	Ligne 102	random_state=42
SVM classifier	pmsm_perspective2_ml.py	Ligne 110	random_state=42
PCA visualization	pmsm_perspective2_ml.py	Ligne 145	random_state=42
Bruit simulation	pmsm_simulation.py	Ligne 45	np.random.seed(42)
Vérification de reproductibilité
bash
# Exécuter 2 fois et comparer les hash des outputs
python scripts/pmsm_health_index.py
md5sum resultats/ch5_health_index_barres.png > hash1.txt
python scripts/pmsm_health_index.py
md5sum resultats/ch5_health_index_barres.png > hash2.txt
diff hash1.txt hash2.txt  # Doit être vide (identique)
📊 Dataset Réel
Source : Bacha et al. (2024)
DOI : 10.5281/zenodo.13974503
Licence : CC BY 4.0
Contenu : 10,892 mesures réelles sur banc d'essai PMSM
Conditions : F0 (normal), F1-F6 (défauts onduleur), F7 (court-circuit statorique), F8 (surchauffe)
Le fichier dataset_reel/converted_dataset.csv est inclus dans ce dépôt pour faciliter la reproduction sans téléchargement externe.
📜 Licence
Ce projet est sous licence MIT.
plain
MIT License

Copyright (c) 2026 Layla Essadiq

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
Données tierces : Le dataset Bacha et al. (2024) est sous licence CC BY 4.0 et reste la propriété de ses auteurs originaux.
🆘 Dépannage
Table
Problème	                    Solution
ModuleNotFoundError         	Vérifier pip install -r requirements.txt dans le venv
FileNotFoundError	            Vérifier que vous êtes à la racine du projet
Figures différentes	          Vérifier Python 3.12.2 exactement, pas 3.11 ou 3.13
Erreur mémoire	              Réduire n_augment à 40 dans pmsm_perspective2_ml.py
Différences de résultats	    Vérifier random_state=42 partout
📧 Contact
Layla Essadiq — layla.essadiq@email.com
Encadrante académique : Pr. Maria Zamzami (ENSAM Rabat)
Encadrante externe : Oumayma Karroum (AFD Technologies)
Dernière mise à jour : 12 juin 2026
Version du pipeline : 1.0.0
