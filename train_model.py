"""
Entraîne le pipeline de prédiction des annulations (Booking AI) avec gestion d'imputation
et équilibrage de classes pour éliminer l'overfitting.
"""

import argparse
import csv
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
    "places_parking_demandees",
    "ville",
]


def load_data(path: str) -> pd.DataFrame:
    if not path.lower().endswith(".csv"):
        return pd.read_excel(path)

    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    seps = (",", ";", "\t")
    last_error = None

    for encoding in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=encoding, sep=sep)
                if df.shape[1] > 1:
                    return df
            except Exception as e:
                last_error = e

    for encoding in encodings:
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                sep=None,
                engine="python",
                on_bad_lines="skip",
                quoting=csv.QUOTE_NONE,
            )
            if df.shape[1] > 1:
                return df
        except Exception as e:
            last_error = e

    raise ValueError(f"Impossible de lire ce CSV : {last_error}")


def build_pipeline(num_cols: list[str]) -> Pipeline:
    # Transformation des variables numériques
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Transformation des variables ordinales
    ord_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    # Transformation des variables nominales
    nom_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Inconnu")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("ord", ord_transformer, ORDINAL_COLS),
            ("nom", nom_transformer, NOMINAL_COLS),
        ],
        remainder="drop",
    )

    # Modèle avec régularisation et équilibrage des classes pour prévenir le biais
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=4,
        class_weight="balanced",  # Résout le problème des prédictions unilatérales
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessing", preprocessor), ("model", model)])


def main(data_path: str, output_path: str) -> None:
    df = load_data(data_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"La colonne cible '{TARGET_COL}' est absente du fichier fourni.")

    drop_cols = [c for c in [TARGET_COL, DATE_COL] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL]

    num_cols = [c for c in X.columns if c not in ORDINAL_COLS + NOMINAL_COLS]

    pipeline = build_pipeline(num_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)

    print("=== Évaluation de la Généralisation ===")
    print(classification_report(y_test, pipeline.predict(X_test)))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output, compress=3)
    print(f"\nPipeline ré-entraîné et sauvegardé avec succès dans : {output.resolve()}")


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
