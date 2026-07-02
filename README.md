# Contrôle gestuel du robot NAO par reconnaissance de posture

Projet Pluridisciplinaire — **M1 Sciences Cognitives, parcours Intelligence Artificielle**

Équipe : PELLARD · BENDINI · ROUAG · ARSLAN · PURSON

---

## Présentation

Ce projet vise à contrôler un robot humanoïde **NAO** par reconnaissance de posture corporelle en temps réel. Un utilisateur effectue un mouvement devant une webcam ; un réseau de neurones classifie ce geste et envoie la commande correspondante au robot simulé dans **Webots**.

L'approche repose sur trois étapes :
1. **Acquisition** : capture des landmarks squelettiques via MediaPipe (pose, mains, visage)
2. **Entraînement** : classification des mouvements par un réseau de neurones dense (Keras)
3. **Contrôle** : le contrôleur Webots lit la posture en direct et pilote le NAO

---

## Mouvements reconnus

| Code | Mouvement |
|------|-----------|
| `a` | Avancer |
| `r` | Reculer |
| `pd` | Pas à droite |
| `pg` | Pas à gauche |
| `pid` | Pivot à droite |
| `pig` | Pivot à gauche |

Chaque mouvement est enregistré à **3 vitesses** (V1 lent → V3 rapide).

---

## Architecture

```
PROJET-PLURIDISCIPLINAIRE/
│
├── dataacquisition.py        # Capture landmarks corporels (MediaPipe Pose) → CSV
├── datacquisition_main.py    # Capture landmarks mains (MediaPipe Hands) → CSV
├── datacquisition_visage.py  # Capture landmarks visage (MediaPipe FaceMesh) → CSV
├── compile_fichier.py        # Fusion de tous les CSV par classe → enfant.csv
├── transforme.py             # Entraînement du réseau de neurones (Keras)
│
├── dataset1/                 # Données d'entraînement par participant et par mouvement
│   ├── <prenom>.csv          # Landmarks bruts par personne
│   └── Avancer/ Reculer/ ... # Sous-dossiers par classe et vitesse
│
├── model/                    # Modèle entraîné (version 1)
│   ├── model.h5              # Réseau de neurones Keras
│   ├── label_encoder.pkl     # Encodeur des étiquettes
│   └── scaler.pkl            # Normaliseur StandardScaler
│
├── model2/                   # Modèle entraîné (version 2)
│
└── nao/                      # Intégration Webots
    ├── controllers/
    │   └── nao_pp_s8/
    │       ├── nao_pp_s8.py  # Contrôleur principal Webots
    │       ├── naomotion.py  # Bibliothèque de mouvements NAO (3 vitesses)
    │       └── pose.py       # Lecture posture temps réel (MediaPipe)
    ├── motions/              # Fichiers .motion NAO (Forwards, Backwards, TurnLeft…)
    └── worlds/               # Scènes Webots
```

---

## Pipeline complet

### 1. Acquisition des données

```bash
# Landmarks corporels (pose)
python dataacquisition.py
# → appuyer sur 'a' pour enregistrer une frame, 'q' pour quitter

# Landmarks mains
python datacquisition_main.py

# Landmarks visage
python datacquisition_visage.py
```

Les CSV sont sauvegardés dans `dataset1/<NomParticipant>/`.

### 2. Compilation du dataset

```bash
cd dataset1/
python compile_fichier.py
# → génère enfant.csv (toutes classes fusionnées avec colonne "action")
```

### 3. Entraînement du modèle

```bash
python transforme.py
# → génère model.h5, label_encoder.pkl, scaler.pkl
```

**Architecture du réseau :**
```
Input (66 features — 33 landmarks × X,Y)
  └─ Dense(64) → LeakyReLU → BatchNorm → Dropout(0.3)
  └─ Dense(32) → LeakyReLU → BatchNorm → Dropout(0.25)
  └─ Dense(num_classes, softmax)
```
- Optimiseur : Adam (lr = 0.0005)
- Loss : categorical_crossentropy
- Epochs : 350, batch_size : 40

### 4. Contrôle du robot dans Webots

1. Copier `model.h5`, `label_encoder.pkl` et `scaler.pkl` dans `nao/controllers/nao_pp_s8/`
2. Ouvrir la scène dans Webots (`nao/worlds/`)
3. Le contrôleur `nao_pp_s8.py` lit la posture en temps réel et fait marcher le NAO

---

## Dataset

- **20+ participants** (enregistrements individuels en CSV)
- **6 mouvements × 3 vitesses** = 18 captures par personne
- Features : coordonnées normalisées (X, Y) de 33 points squelettiques MediaPipe

---

## Dépendances

```bash
pip install mediapipe opencv-python tensorflow scikit-learn pandas numpy joblib matplotlib
```

Webots doit être installé séparément : [cyberbotics.com](https://cyberbotics.com)

---

## Contexte académique

Projet réalisé dans le cadre du **Projet Pluridisciplinaire de M1 Sciences Cognitives**, parcours **Intelligence Artificielle**, à l'Université Paris Cité.

L'objectif pédagogique est d'intégrer des méthodes d'apprentissage automatique (traitement du signal corporel, classification multiclasse) dans un système de contrôle robotique incarné, en lien avec les thématiques de **cognition embodiée** et d'**interaction humain-robot**.
