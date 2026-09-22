"""Booking AI — application Streamlit."""

import re
from io import BytesIO
from pathlib import Path

import csv
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Booking AI", page_icon="🏨", layout="wide")

TARGET_COL = "reservation_annulee"
DATE_COL = "date_arrivee"
# Proxy pour le "lieu d'origine de la réservation" : aucune colonne pays/lieu
# n'existe dans le schéma fourni. Remplacer par la vraie colonne dès qu'elle
# est disponible (ex. "pays_origine").
LOCATION_COL = "canal_distribution"

MODEL_PATH = Path(__file__).parent / "model" / "booking_ai_pipeline.joblib"
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def load_data(uploaded_file) -> pd.DataFrame:
    """Charge un fichier CSV/Excel avec contrôles de base (taille, extension)."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Extension non autorisée : {suffix}")
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"Fichier trop volumineux (> {MAX_FILE_SIZE_MB} Mo)")

    if suffix != ".csv":
        return pd.read_excel(uploaded_file, engine="openpyxl")

    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    seps = (",", ";", "\t")
    last_error = None

    for encoding in encodings:
        for sep in seps:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding, sep=sep)
                if df.shape[1] > 1:
                    return df
            except Exception as e:  # noqa: BLE001
                last_error = e

    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(
                uploaded_file,
                encoding=encoding,
                sep=None,
                engine="python",
                on_bad_lines="skip",
                quoting=csv.QUOTE_NONE,
            )
            if df.shape[1] > 1:
                return df
        except Exception as e:  # noqa: BLE001
            last_error = e

    raise ValueError(f"Impossible de lire ce CSV (encodage/séparateur non reconnu) : {last_error}")


def iqr_bounds(series: pd.Series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def sanitize_for_excel(value):
    """Empêche l'injection de formules dans le fichier Excel exporté."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@"):
        return "'" + value
    return value


# ---------------------------------------------------------------------------
# Onglets
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    ["🏨 Carte d'identité", "📊 Exploration & Dashboards", "🤖 Prédiction"]
)

# --- Onglet 1 ---------------------------------------------------------------
with tab1:
    st.title("Booking AI")
    st.subheader(
        "De la conception au meilleur choix stratégique pour votre "
        "accompagnement touristique."
    )
    st.markdown(
        """
        **Objectif de la plateforme**

        Booking AI accompagne les établissements touristiques dans leurs
        démarches :
        - Suivi des indicateurs clés de performance (KPI)
        - Analyse de la rentabilité
        - Prédiction des annulations de réservations grâce au Machine Learning
        """
    )

# --- Onglet 2 ---------------------------------------------------------------
with tab2:
    st.header("Exploration des données")
    file2 = st.file_uploader(
        "Importer un fichier (Excel ou CSV)", type=["csv", "xlsx"], key="explore"
    )

    if file2:
        try:
            df = load_data(file2)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        if DATE_COL in df.columns:
            df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

        st.subheader("Aperçu de la base de données")
        st.dataframe(df.head(20))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lignes", len(df))
        c2.metric("Variables", df.shape[1])
        c3.metric("Doublons", int(df.duplicated().sum()))
        c4.metric("Valeurs manquantes", int(df.isna().sum().sum()))

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Types de variables**")
            st.dataframe(
                df.dtypes.astype(str).value_counts()
                .rename("nombre").rename_axis("type")
            )
        with col_b:
            st.markdown("**Valeurs manquantes par colonne**")
            na = df.isna().sum()
            na = na[na > 0]
            st.dataframe(na.rename("valeurs manquantes") if len(na) else pd.DataFrame({"valeurs manquantes": []}))

        st.subheader("Valeurs aberrantes (méthode IQR)")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        rows = []
        for col in num_cols:
            lower, upper = iqr_bounds(df[col])
            mask = (df[col] < lower) | (df[col] > upper)
            rows.append(
                {
                    "variable": col,
                    "nb_outliers": int(mask.sum()),
                    "borne_basse": round(lower, 2),
                    "borne_haute": round(upper, 2),
                }
            )
        outlier_df = pd.DataFrame(rows).sort_values("nb_outliers", ascending=False)
        st.dataframe(outlier_df, use_container_width=True)

        if num_cols:
            selected_col = st.selectbox("Détail par variable numérique", num_cols)
            lower, upper = iqr_bounds(df[selected_col])
            mask = (df[selected_col] < lower) | (df[selected_col] > upper)
            st.plotly_chart(
                px.box(df, y=selected_col, points="outliers", title=f"Boxplot — {selected_col}"),
                use_container_width=True,
            )
            st.caption(f"{int(mask.sum())} valeur(s) aberrante(s) détectée(s)")
            st.dataframe(df[mask])

        st.subheader("Statistiques descriptives")
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

        if len(num_cols) > 1:
            st.subheader("Matrice de corrélation")
            corr = df[num_cols].corr()
            st.plotly_chart(
                px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation map"),
                use_container_width=True,
            )

        st.subheader("Tableaux de bord décisionnels")
        f1, f2 = st.columns(2)
        date_range = None
        if DATE_COL in df.columns and df[DATE_COL].notna().any():
            min_d, max_d = df[DATE_COL].min(), df[DATE_COL].max()
            date_range = f1.date_input("Filtrer par date", (min_d, max_d))
        location_filter = None
        if LOCATION_COL in df.columns:
            options = ["Tous"] + sorted(df[LOCATION_COL].dropna().unique().tolist())
            location_filter = f2.selectbox("Filtrer par lieu / canal de distribution", options)

        dff = df.copy()
        if date_range and len(date_range) == 2 and DATE_COL in dff.columns:
            dff = dff[
                (dff[DATE_COL] >= pd.to_datetime(date_range[0]))
                & (dff[DATE_COL] <= pd.to_datetime(date_range[1]))
            ]
        if location_filter and location_filter != "Tous":
            dff = dff[dff[LOCATION_COL] == location_filter]

        if TARGET_COL in dff.columns:
            status_map = {0: "Confirmée", 1: "Annulée"}
            dff = dff.copy()
            dff["_statut"] = dff[TARGET_COL].map(status_map).fillna(dff[TARGET_COL].astype(str))

            colA, colB = st.columns(2)
            with colA:
                st.plotly_chart(
                    px.pie(dff, names="_statut", title="Réservations annulées vs confirmées (total)"),
                    use_container_width=True,
                )
            if "type_hotel" in dff.columns:
                with colB:
                    by_hotel = dff.groupby(["type_hotel", "_statut"]).size().reset_index(name="nb")
                    st.plotly_chart(
                        px.bar(by_hotel, x="type_hotel", y="nb", color="_statut", barmode="group", title="Par hôtel"),
                        use_container_width=True,
                    )

            if DATE_COL in dff.columns and dff[DATE_COL].notna().any():
                by_date = (
                    dff.dropna(subset=[DATE_COL])
                    .groupby([pd.Grouper(key=DATE_COL, freq="MS"), "_statut"])
                    .size()
                    .reset_index(name="nb")
                )
                st.plotly_chart(
                    px.line(by_date, x=DATE_COL, y="nb", color="_statut", title="Évolution dans le temps"),
                    use_container_width=True,
                )

            if LOCATION_COL in dff.columns:
                by_loc = dff.groupby([LOCATION_COL, "_statut"]).size().reset_index(name="nb")
                st.plotly_chart(
                    px.bar(by_loc, x=LOCATION_COL, y="nb", color="_statut", barmode="group", title="Par lieu / canal de distribution"),
                    use_container_width=True,
                )
        else:
            st.info("La colonne cible n'est pas présente : les graphiques annulée/confirmée ne peuvent pas être affichés pour ce fichier.")

        k1, k2 = st.columns(2)
        if "client_nouveau" in dff.columns:
            k1.metric("Nouveaux clients", int(dff["client_nouveau"].astype(bool).sum()))
        if "client_fidele" in dff.columns:
            k2.metric("Clients fidèles", int(dff["client_fidele"].astype(bool).sum()))

        if "segment_marche" in dff.columns:
            st.plotly_chart(
                px.pie(dff, names="segment_marche", title="Répartition par segment de marché"),
                use_container_width=True,
            )

# --- Onglet 3 ---------------------------------------------------------------
with tab3:
    st.header("Prédiction des annulations")
    st.caption(
        "Le modèle doit être entraîné au préalable avec `train_model.py` sur "
        "un jeu de données historique labellisé. Importez ici un fichier "
        "SANS la variable cible pour obtenir une prédiction."
    )

# --- Onglet 3 ---------------------------------------------------------------
with tab3:
    st.header("Prédiction des annulations")
    st.caption(
        "Importez un fichier SANS la variable cible pour obtenir une "
        "prédiction (le modèle a déjà été entraîné en amont)."
    )

    if not MODEL_PATH.exists():
        st.error(
            "Fichier modèle introuvable (`model/booking_ai_pipeline.joblib`). "
            "Ajoutez-le au dépôt pour activer cet onglet."
        )
    else:
        pipeline = joblib.load(MODEL_PATH)
        file3 = st.file_uploader(
            "Importer le fichier à prédire (Excel ou CSV)", type=["csv", "xlsx"], key="predict"
        )

        if file3:
            try:
                df_pred = load_data(file3)
            except ValueError as e:
                st.error(str(e))
                st.stop()

            dates = df_pred[DATE_COL] if DATE_COL in df_pred.columns else pd.Series([pd.NA] * len(df_pred))
            X = df_pred.drop(columns=[c for c in [TARGET_COL, DATE_COL] if c in df_pred.columns])

            try:
                preds = pipeline.predict(X)
            except Exception as e:
                st.error(
                    "Erreur lors de la prédiction — vérifiez que les colonnes du "
                    f"fichier correspondent au schéma attendu.\n\nDétail technique : {e}"
                )
                st.stop()

            result = pd.DataFrame(
                {
                    DATE_COL: dates.values,
                    "statut_predit": np.where(preds == 1, "Annulée", "Confirmée"),
                }
            )
            result = result.applymap(sanitize_for_excel)

            n_confirmees = int((preds == 0).sum())
            n_annulees = int((preds == 1).sum())

            c1, c2 = st.columns(2)
            c1.metric("Réservations prédites confirmées", n_confirmees)
            c2.metric("Réservations prédites annulées", n_annulees)

            st.dataframe(result, use_container_width=True)

            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                result.to_excel(writer, index=False, sheet_name="predictions")
            st.download_button(
                "📥 Télécharger les prédictions (Excel)",
                data=buffer.getvalue(),
                file_name="predictions_booking_ai.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
