import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import joblib
import matplotlib.pyplot as plt

# --- Charger les données ---
# ⚠️ Remplace ça par le chargement de ton fichier de données réel
# Exemple : df = pd.read_csv("poses.csv")
df = pd.read_csv("/Users/onurarslan/Desktop/robot/controllers/nao_demo_python/enfant.csv", encoding="latin1")
df.columns = df.columns.str.strip()  # nettoyage des noms de colonnes
print("Colonnes disponibles :", df.columns.tolist())

# --- Préparer X et y ---
y = df["action"]
X = df.drop(columns=["action"])

# Encodage des labels (single-label)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
y_train_cat = to_categorical(y_encoded, num_classes)

# Split des données
X_train_clean, X_test_clean, y_train_encoded, y_test_encoded = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test_clean)

# One-hot encoding des étiquettes
y_train_cat = to_categorical(y_train_encoded, num_classes)
y_test_cat = to_categorical(y_test_encoded, num_classes)

# --- Définir le modèle ---
model = Sequential([
    Dense(64, input_shape=(X_train_scaled.shape[1],), kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.25),

    Dense(num_classes, activation='softmax')
])

# Compilation
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement
history = model.fit(
    X_train_scaled, y_train_cat,
    epochs=350,
    batch_size=40,
    validation_split=0.2,
    verbose=1
)

# --- Sauvegarde du modèle et des objets ---
model.save("model.h5")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Modèle et fichiers sauvegardés avec succès.")
