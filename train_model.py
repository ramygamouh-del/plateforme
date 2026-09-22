"""
Entraîne le pipeline de prédiction des annulations (Booking AI) et le sauvegarde.

Usage :
    python train_model.py chemin/vers/dataset_labellise.csv
    python train_model.py chemin/vers/dataset_labellise.xlsx --output model/booking_ai_pipeline.joblib

Le dataset fourni ici DOIT contenir la colonne cible `reservation_annulee`
(c'est le jeu de données historique servant à l'apprentissage, différent du
fichier "à prédire" utilisé ensuite dans l'onglet 3 de l'application).
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET_COL = "reservation_annulee"
DATE_COL = "date_arrivee"

ORDINAL_COLS = [
    "nombre_adultes",
    "nombre_enfants",
    "nombre_bebes",
    "nuits_weekend",
    "nuits_semaine",
    "annulations_precedentes",
    "reservations_precedentes_non_annulees",
    "nombre_demandes_speciales",
]

NOMINAL_COLS = [
    "client_fidele",
    "type_repas",
    "type_chambre_reservee",
    "type_chambre_attribuee",
    "type_depot",
    "segment_marche",
    "canal_distribution",
    "type_hotel",
    "client_nouveau",
    "place_parking_requise",
]


def load_data(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def build_pipeline(num_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "ord",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ORDINAL_COLS,
            ),
            (
                "nom",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                NOMINAL_COLS,
            ),
        ],
        remainder="drop",
    )
    model = RandomForestClassifier(random_state=42, n_estimators=300, n_jobs=-1)
    return Pipeline(steps=[("preprocessing", preprocessor), ("model", model)])


def main(data_path: str, output_path: str) -> None:
    df = load_data(data_path)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"La colonne cible '{TARGET_COL}' est absente du fichier fourni. "
            "Ce script attend un jeu de données HISTORIQUE labellisé pour l'entraînement."
        )

    drop_cols = [c for c in [TARGET_COL, DATE_COL] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL]

    num_cols = [c for c in X.columns if c not in ORDINAL_COLS + NOMINAL_COLS]

    pipeline = build_pipeline(num_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)

    print("=== Évaluation sur le jeu de test ===")
    print(classification_report(y_test, pipeline.predict(X_test)))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output)
    print(f"\nPipeline sauvegardé dans : {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", help="Chemin du dataset labellisé (CSV ou Excel)")
    parser.add_argument(
        "--output",
        default="model/booking_ai_pipeline.joblib",
        help="Chemin de sortie du pipeline entraîné",
    )
    args = parser.parse_args()
    main(args.data_path, args.output)
