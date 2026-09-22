# Booking AI

Plateforme Streamlit d'accompagnement des établissements touristiques :
carte d'identité, exploration/dashboards décisionnels, et prédiction des
annulations de réservations par Machine Learning.

## Structure du projet
```
booking-ai/
├── app.py              # application Streamlit (3 onglets)
├── train_model.py      # script d'entraînement du modèle
├── requirements.txt
├── model/              # pipeline entraîné (généré par train_model.py)
└── README.md
```

## 1. Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Entraîner le modèle (une seule fois, ou à chaque mise à jour des données)

Il vous faut un jeu de données **historique et labellisé** (qui contient la
colonne `reservation_annulee`) — par exemple un export de réservations
passées. Ce n'est **pas** le même fichier que celui que l'on importera
ensuite pour prédire.

```bash
python train_model.py chemin/vers/dataset_historique.csv
```

Cela génère `model/booking_ai_pipeline.joblib`, utilisé automatiquement par
l'onglet 3 de l'application. Le rapport de classification (précision,
rappel, f1-score) s'affiche dans le terminal — vérifiez qu'il est
satisfaisant avant de mettre le modèle en production.

## 3. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`.

- **Onglet 1** : présentation de la plateforme (statique).
- **Onglet 2** : importez n'importe quel fichier CSV/Excel de réservations
  pour explorer les données, détecter les valeurs aberrantes (IQR),
  consulter les statistiques descriptives, la matrice de corrélation et les
  dashboards décisionnels (filtrables par date et par canal/lieu).
- **Onglet 3** : importez un fichier **sans** la colonne cible pour obtenir
  les prédictions (fichier Excel téléchargeable + compte rendu).

## 4. Déploiement

### Option rapide — Streamlit Community Cloud
1. Poussez ce dossier dans un dépôt GitHub (incluez le fichier `model/*.joblib` entraîné, ou générez-le via une étape CI).
2. Connectez le dépôt sur [share.streamlit.io](https://share.streamlit.io).
3. HTTPS est géré automatiquement.

### Option maîtrisée — Docker + reverse proxy
Exemple de `Dockerfile` minimal :
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```
Placez ensuite un reverse proxy (Nginx/Traefik) devant le conteneur pour
gérer le certificat TLS (HTTPS) et, si nécessaire, l'authentification.

## 5. Points de sécurité déjà couverts dans le code
- Validation de l'extension et de la taille des fichiers importés.
- Neutralisation des cellules pouvant contenir une formule (`=`, `+`, `-`,
  `@`) avant export Excel (protection contre le CSV/Excel injection).
- Aucun contenu de fichier n'est journalisé.
- Chargement du modèle via `joblib` : ne jamais remplacer
  `model/booking_ai_pipeline.joblib` par un fichier provenant d'une source
  non fiable (risque d'exécution de code à la désérialisation).

Pour l'authentification des utilisateurs et le HTTPS, voir la section
"Sécurité" du cahier des charges.

## Remarque
La colonne `canal_distribution` est utilisée comme **proxy** du "lieu
d'origine de la réservation" tant qu'aucune colonne pays/lieu explicite
n'existe dans le schéma. Remplacez `LOCATION_COL` dans `app.py` dès qu'une
colonne dédiée (ex. `pays_origine`) est disponible.
