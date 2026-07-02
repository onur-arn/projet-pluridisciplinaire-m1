import os
import pandas as pd

# Mapping dossier → code action
action_map = {
    "Avancer": "a",
    "Reculer": "r",
    "Pas_D": "pd",
    "Pas_G": "pg",
    "Pivot_D": "pid",
    "Pivot_G": "pig",
}

# Ordre des actions dans les fichiers "personne"
ordered_actions = [
    ('a', 1), ('a', 2), ('a', 3),
    ('r', 1), ('r', 2), ('r', 3),
    ('pg', 1), ('pg', 2), ('pg', 3),
    ('pd', 1), ('pd', 2), ('pd', 3),
    ('pig', 1), ('pig', 2), ('pig', 3),
    ('pid', 1), ('pid', 2), ('pid', 3),
]

DEFAULT_FRAMES_PER_STEP = None  # ou un entier (ex: 30) si tu veux forcer une taille

all_dfs = []

### --- Partie 1 : fichiers de structure dossier courant / Action / Vitesse / *.csv ---
for action_dir, action_code in action_map.items():
    for vitesse in range(1, 4):
        path = os.path.join(os.getcwd(), action_dir, f"V{vitesse}")
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith(".csv"):
                    full_path = os.path.join(path, file)
                    try:
                        df = pd.read_csv(full_path)
                        df["action"] = action_code+f"_{vitesse}"
                        all_dfs.append(df)
                    except Exception as e:
                        print(f"❌ Erreur lecture fichier: {full_path} → {e}")

### --- Partie 2 : fichiers "personne" dans dossier courant (nom.csv) ---
for file in os.listdir():
    if file.endswith(".csv") and not file.startswith("merged") and os.path.isfile(file):
        skip = False
        # On vérifie que ce n'est pas un fichier dans un sous-dossier d’action
        for action_folder in action_map.keys():
            if os.path.commonpath([os.path.abspath(file), os.path.abspath(action_folder)]) == os.path.abspath(action_folder):
                skip = True
                break
        if skip:
            continue

        try:
            df = pd.read_csv(file)
            total = len(df)
            steps = len(ordered_actions)

            if DEFAULT_FRAMES_PER_STEP:
                step_size = DEFAULT_FRAMES_PER_STEP
            else:
                step_size = total // steps if steps else total

            idx = 0
            for action_code, vitesse in ordered_actions:
                if idx >= total:
                    break
                next_idx = idx + step_size
                part = df.iloc[idx:next_idx].copy()
                part["action"] =  action_code+f"_{vitesse}"
                all_dfs.append(part)
                idx = next_idx

        except Exception as e:
            print(f"❌ Erreur traitement fichier personne: {file} → {e}")

### --- Fusion et export final ---
if all_dfs:
    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df.to_csv("enfant.csv", index=False)
    print("✅ Tous les fichiers fusionnés dans 'enfant.csv'")
else:
    print("⚠️ Aucun fichier fusionné.")