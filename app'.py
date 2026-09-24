"""Booking AI — Application Streamlit révisée avec navigation latérale et rapports PDF."""

import csv
from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Importation de ReportLab pour la génération de rapports PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

st.set_page_config(page_title="Booking AI", page_icon="🏨", layout="wide")

TARGET_COL = "reservation_annulee"
DATE_COL = "date_arrivee"
LOCATION_COL = "ville"
CHANNEL_COL = "canal_distribution"
HOTEL_COL = "type_hotel"

MODEL_PATH = Path(__file__).parent / "model" / "booking_ai_pipeline.joblib"
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}


# ---------------------------------------------------------------------------
# Utilitaire : Chargement et Nettoyage des Données
# ---------------------------------------------------------------------------

@st.cache_resource
def load_trained_pipeline(model_path: Path):
    """Charge le modèle joblib avec gestion des erreurs d'incompatibilité."""
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.warning(f"⚠️ Incompatibilité avec le fichier modèle : {e}")
            return None
    return None


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
            except Exception as e:
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
        except Exception as e:
            last_error = e

    raise ValueError(f"Impossible de lire ce CSV (encodage/séparateur non reconnu) : {last_error}")


def iqr_bounds(series: pd.Series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def sanitize_for_excel(value):
    """Empêche l'injection de formules dans l'export Excel."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@"):
        return "'" + value
    return value


# ---------------------------------------------------------------------------
# Utilitaire : Génération des Rapports PDF
# ---------------------------------------------------------------------------

def generate_pdf_report(title: str, subtitle: str, summary_data: list, table_data: list) -> bytes:
    """Génère un rapport PDF élégant structuré avec ReportLab."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=6,
    )
    sub_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=15,
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#1F2937"),
        spaceBefore=12,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#374151"),
        spaceAfter=6,
    )

    story = [
        Paragraph(f"<b>{title}</b>", title_style),
        Paragraph(f"{subtitle} — Édité le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", sub_style),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=15),
    ]

    # Section Synthèse
    story.append(Paragraph("<b>Résumé des indicateurs principaux :</b>", section_style))
    for item in summary_data:
        story.append(Paragraph(f"• <b>{item['label']} :</b> {item['value']}", body_style))

    story.append(Spacer(1, 15))

    # Tableau
    if table_data:
        story.append(Paragraph("<b>Aperçu synthétique des données :</b>", section_style))
        t = Table(table_data, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("TOPPADDING", (0, 0), (-1, 0), 6),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F9FAFB")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                ]
            )
        )
        story.append(t)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Navigation Latérale (Sidebar)
# ---------------------------------------------------------------------------

st.sidebar.title("🏨 Booking AI")
st.sidebar.markdown("---")

menu_option = st.sidebar.radio(
    " Navigation Principale",
    [
        "📌 Présentation & Vue d'ensemble",
        "📊 Exploration & Dashboards",
        "🤖 Prédiction des annulations",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Booking AI v2.0 • Plateforme Décisionnelle")


# ===========================================================================
# Section 1 : Présentation & Vue d'ensemble
# ===========================================================================
if menu_option == "📌 Présentation & Vue d'ensemble":
    st.title("📌 Présentation & Vue d'ensemble")
    st.subheader("De la donnée brute aux décisions stratégiques touristiques.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
            Welcome sur **Booking AI**, votre solution d'intelligence décisionnelle dédiée à l'hôtellerie et à l'accompagnement touristique.

            ### 🎯 Objectifs de la plateforme :
            1. **Pilotage & KPI** : Suivi rigoureux des indicateurs de performance hôtelière.
            2. **Analyse Exploratoire** : Audit de la qualité des données, statistiques descriptives poussées et détection d'anomalies.
            3. **Prédiction IA des Annulations** : Anticipation des annulations de réservation via Machine Learning pour optimiser le Yield Management.
            """
        )
    with col2:
        st.info(
            """
            💡 **Guide rapide :**
            - Rendez-vous sur **Exploration & Dashboards** pour importer et analyser vos fichiers clients.
            - Utilisez l'onglet **Prédiction** pour soumettre vos réservations futures et anticiper le risque d'annulation.
            """
        )


# ===========================================================================
# Section 2 : Exploration et Dashboards
# ===========================================================================
elif menu_option == "📊 Exploration & Dashboards":
    st.header("📊 Exploration des données & Dashboards")

    file2 = st.file_uploader("Importer un fichier (Excel ou CSV)", type=["csv", "xlsx"], key="explore")

    if file2:
        try:
            df = load_data(file2)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        if DATE_COL in df.columns:
            df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

        st.subheader("Aperçu de la base de données")
        st.dataframe(df.head(15), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lignes", len(df))
        c2.metric("Variables", df.shape[1])
        c3.metric("Doublons", int(df.duplicated().sum()))
        c4.metric("Valeurs manquantes", int(df.isna().sum().sum()))

        st.markdown("---")
        st.subheader("📌 Statistiques Descriptives Séparées")

        num_df = df.select_dtypes(include=np.number)
        cat_df = df.select_dtypes(include=["object", "category"])
        date_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()

        tab_num, tab_cat, tab_date = st.tabs(
            ["🔢 Variables Numériques", "🔤 Variables Catégorielles", "📅 Variables Temporelles"]
        )

        with tab_num:
            if not num_df.empty:
                st.dataframe(num_df.describe().T, use_container_width=True)
            else:
                st.info("Aucune variable numérique détectée.")

        with tab_cat:
            if not cat_df.empty:
                st.dataframe(cat_df.describe(include="all").T, use_container_width=True)
            else:
                st.info("Aucune variable catégorielle détectée.")

        with tab_date:
            if date_cols:
                date_stats = []
                for col in date_cols:
                    s = df[col].dropna()
                    date_stats.append(
                        {
                            "Variable": col,
                            "Date min": s.min().strftime("%Y-%m-%d") if not s.empty else "N/A",
                            "Date max": s.max().strftime("%Y-%m-%d") if not s.empty else "N/A",
                            "Plage (jours)": (s.max() - s.min()).days if not s.empty else 0,
                            "Valeurs manquantes": int(df[col].isna().sum()),
                        }
                    )
                st.dataframe(pd.DataFrame(date_stats), use_container_width=True)
            else:
                st.info("Aucune colonne de type Date identifiée.")

        # Outliers & Corrélations
        st.markdown("---")
        col_out, col_corr = st.columns(2)

        with col_out:
            with st.container(border=True):
                st.subheader("Valeurs aberrantes (Méthode IQR)")
                num_cols = num_df.columns.tolist()
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
                st.dataframe(outlier_df, height=220, use_container_width=True)

        with col_corr:
            with st.container(border=True):
                st.subheader("Matrice de Corrélation Clarifiée")
                if len(num_cols) > 1:
                    corr = df[num_cols].corr()
                    fig_corr = px.imshow(
                        corr,
                        text_auto=".2f",
                        color_continuous_scale="Blues",
                        aspect="auto",
                        title="Corrélations Linéaires",
                    )
                    fig_corr.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Variables numériques insuffisantes pour la matrice de corrélation.")

        # Dashboards et KPI
        st.markdown("---")
        st.subheader("📈 Tableaux de bord décisionnels & KPI")

        # Filtres KPI à 4 critères
        st.markdown("**🎛️ Filtres combinés :**")
        fk1, fk2, fk3, fk4 = st.columns(4)

        date_range = None
        if DATE_COL in df.columns and df[DATE_COL].notna().any():
            min_d, max_d = df[DATE_COL].min(), df[DATE_COL].max()
            date_range = fk1.date_input("1. Filtrer par date", (min_d, max_d))

        hotel_filter = "Tous"
        if HOTEL_COL in df.columns:
            hotel_opts = ["Tous"] + sorted(df[HOTEL_COL].dropna().unique().tolist())
            hotel_filter = fk2.selectbox("2. Type d'hôtel", hotel_opts)

        loc_filter = "Tous"
        if LOCATION_COL in df.columns:
            loc_opts = ["Tous"] + sorted(df[LOCATION_COL].dropna().unique().tolist())
            loc_filter = fk3.selectbox("3. Lieu / Ville", loc_opts)

        chan_filter = "Tous"
        if CHANNEL_COL in df.columns:
            chan_opts = ["Tous"] + sorted(df[CHANNEL_COL].dropna().unique().tolist())
            chan_filter = fk4.selectbox("4. Canal de distribution", chan_opts)

        # Application des filtres
        dff = df.copy()
        if date_range and len(date_range) == 2 and DATE_COL in dff.columns:
            dff = dff[
                (dff[DATE_COL] >= pd.to_datetime(date_range[0]))
                & (dff[DATE_COL] <= pd.to_datetime(date_range[1]))
            ]
        if hotel_filter != "Tous" and HOTEL_COL in dff.columns:
            dff = dff[dff[HOTEL_COL] == hotel_filter]
        if loc_filter != "Tous" and LOCATION_COL in dff.columns:
            dff = dff[dff[LOCATION_COL] == loc_filter]
        if chan_filter != "Tous" and CHANNEL_COL in dff.columns:
            dff = dff[dff[CHANNEL_COL] == chan_filter]

        # Visualisations encadrées dans des containers
        g1, g2 = st.columns(2)
        if TARGET_COL in dff.columns:
            status_map = {0: "Confirmée", 1: "Annulée"}
            dff["_statut"] = dff[TARGET_COL].map(status_map).fillna(dff[TARGET_COL].astype(str))

            with g1:
                with st.container(border=True):
                    fig_pie = px.pie(
                        dff,
                        names="_statut",
                        title="Répartition des Annulations (Filtre actif)",
                        color_discrete_sequence=["#10B981", "#EF4444"],
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

            with g2:
                with st.container(border=True):
                    if HOTEL_COL in dff.columns:
                        by_hotel = dff.groupby([HOTEL_COL, "_statut"]).size().reset_index(name="nb")
                        fig_bar_h = px.bar(
                            by_hotel,
                            x=HOTEL_COL,
                            y="nb",
                            color="_statut",
                            barmode="group",
                            title="Réservations par Type d'Hôtel",
                        )
                        st.plotly_chart(fig_bar_h, use_container_width=True)
                    else:
                        st.info("Colonne 'type_hotel' absente.")

            g3, g4 = st.columns(2)
            with g3:
                with st.container(border=True):
                    if LOCATION_COL in dff.columns:
                        by_loc = dff.groupby([LOCATION_COL, "_statut"]).size().reset_index(name="nb")
                        fig_loc = px.bar(
                            by_loc,
                            x=LOCATION_COL,
                            y="nb",
                            color="_statut",
                            barmode="group",
                            title="Réservations par Ville / Lieu",
                        )
                        st.plotly_chart(fig_loc, use_container_width=True)

            with g4:
                with st.container(border=True):
                    if CHANNEL_COL in dff.columns:
                        by_chan = dff.groupby([CHANNEL_COL, "_statut"]).size().reset_index(name="nb")
                        fig_chan = px.bar(
                            by_chan,
                            x=CHANNEL_COL,
                            y="nb",
                            color="_statut",
                            barmode="group",
                            title="Réservations par Canal de Distribution",
                        )
                        st.plotly_chart(fig_chan, use_container_width=True)

        # Rapport PDF Exploration
        st.markdown("---")
        st.subheader("📄 Exportation du Rapport d'Exploration")

        summary_pdf = [
            {"label": "Nombre total d'enregistrements (filtrés)", "value": str(len(dff))},
            {"label": "Période analysée", "value": f"{date_range[0]} au {date_range[1]}" if date_range else "Globale"},
            {"label": "Filtre Hôtel", "value": hotel_filter},
            {"label": "Filtre Ville", "value": loc_filter},
            {"label": "Filtre Canal", "value": chan_filter},
        ]

        if TARGET_COL in dff.columns:
            nb_ann = int((dff[TARGET_COL] == 1).sum())
            taux_ann = round((nb_ann / len(dff) * 100), 2) if len(dff) > 0 else 0
            summary_pdf.append({"label": "Taux d'annulation", "value": f"{taux_ann}% ({nb_ann} annulations)"})

        table_data_pdf = [["Métrique", "Valeur"]] + [[s["label"], s["value"]] for s in summary_pdf]

        pdf_bytes = generate_pdf_report(
            title="Rapport d'Exploration & Dashboards",
            subtitle="Booking AI — Analyse Décisionnelle",
            summary_data=summary_pdf,
            table_data=table_data_pdf,
        )

        st.download_button(
            "📥 Télécharger le rapport d'exploration (PDF)",
            data=pdf_bytes,
            file_name=f"rapport_exploration_booking_ai_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
        )


# ===========================================================================
# Section 3 : Prédiction des annulations
# ===========================================================================
elif menu_option == "🤖 Prédiction des annulations":
    st.header("🤖 Prédiction des annulations")
    st.caption("Importez un fichier sans variable cible pour obtenir les prédictions du modèle entraîné.")

    pipeline = load_trained_pipeline(MODEL_PATH)

    if pipeline is None:
        st.error(
            "Fichier modèle introuvable (`model/booking_ai_pipeline.joblib`). "
            "Veuillez exécuter `train_model.py` au préalable."
        )
    else:
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
                # Récupération des probabilités si le modèle les fournit
                if hasattr(pipeline, "predict_proba"):
                    probs = pipeline.predict_proba(X)[:, 1]
                else:
                    probs = None
                    preds_default = pipeline.predict(X)
            except Exception as e:
                st.error(
                    "Erreur lors de la prédiction — vérifiez la compatibilité des colonnes du fichier.\n\n"
                    f"Détail technique : {e}"
                )
                st.stop()

            # Réglage dynamique du seuil de décision
            st.markdown("---")
            st.subheader("⚙️ Paramétrage du seuil de détection du risque")
            
            threshold = st.slider(
                "Seuil de probabilité pour classer une réservation comme 'Annulée'",
                min_value=0.10,
                max_value=0.90,
                value=0.50,
                step=0.05,
                help="Si la probabilité d'annulation dépasse ce seuil, la réservation sera prédite comme 'Annulée'."
            )

            # Application du seuil selon la probabilité
            if probs is not None:
                preds = np.where(probs >= threshold, 1, 0)
            else:
                preds = preds_default

            # Structure du DataFrame de résultat
            result = pd.DataFrame(
                {
                    DATE_COL: dates.values,
                    "statut_predit": np.where(preds == 1, "Annulée", "Confirmée"),
                }
            )
            if probs is not None:
                result["probabilite_annulation_%"] = (probs * 100).round(2)

            result = result.map(sanitize_for_excel) if hasattr(result, "map") else result.applymap(sanitize_for_excel)

            n_confirmees = int((preds == 0).sum())
            n_annulees = int((preds == 1).sum())
            total_preds = len(preds)

            st.markdown("---")
            st.subheader("📊 Statistiques des Réservations Prédites")

            m1, m2, m3 = st.columns(3)
            m1.metric("Total réservations prédites", total_preds)
            m2.metric("Prédites Confirmées 🟢", n_confirmees)
            m3.metric("Prédites Annulées 🔴", n_annulees)

            # Construction sécurisée du DataFrame pour le camembert
            pred_summary = pd.DataFrame(
                {
                    "Statut": ["Confirmée", "Annulée"],
                    "Nombre": [n_confirmees, n_annulees]
                }
            )

            # Graphiques des prédictions dans des conteneurs bordurés
            cp1, cp2 = st.columns(2)
            with cp1:
                with st.container(border=True):
                    if total_preds > 0:
                        fig_pred_pie = px.pie(
                            pred_summary,
                            names="Statut",
                            values="Nombre",
                            title="Répartition Prédictive des Annulations",
                            color="Statut",
                            color_discrete_map={"Confirmée": "#10B981", "Annulée": "#EF4444"},
                            hole=0.3
                        )
                        st.plotly_chart(fig_pred_pie, use_container_width=True)
                    else:
                        st.info("Aucune donnée à afficher.")

            with cp2:
                with st.container(border=True):
                    if probs is not None:
                        fig_hist = px.histogram(
                            result,
                            x="probabilite_annulation_%",
                            nbins=20,
                            title="Distribution des Probabilités d'Annulation (%)",
                            color_discrete_sequence=["#3B82F6"],
                        )
                        # Ligne verticale indiquant le seuil actuel
                        fig_hist.add_vline(
                            x=threshold * 100, 
                            line_dash="dash", 
                            line_color="red", 
                            annotation_text=f"Seuil ({int(threshold*100)}%)"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    else:
                        st.info("Les probabilités ne sont pas disponibles pour ce modèle.")

            st.subheader("Tableau des résultats prédits")
            st.dataframe(result, use_container_width=True)

            # Boutons de Téléchargement (Excel + PDF)
            col_d1, col_d2 = st.columns(2)

            with col_d1:
                buffer_excel = BytesIO()
                with pd.ExcelWriter(buffer_excel, engine="openpyxl") as writer:
                    result.to_excel(writer, index=False, sheet_name="predictions")
                st.download_button(
                    "📥 Télécharger les prédictions (Excel)",
                    data=buffer_excel.getvalue(),
                    file_name="predictions_booking_ai.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            with col_d2:
                pct_conf = (n_confirmees / total_preds * 100) if total_preds > 0 else 0
                pct_ann = (n_annulees / total_preds * 100) if total_preds > 0 else 0

                summary_pred_pdf = [
                    {"label": "Volume total analysé", "value": str(total_preds)},
                    {"label": "Seuil de décision appliqué", "value": f"{int(threshold * 100)}%"},
                    {"label": "Réservations Confirmées", "value": f"{n_confirmees} ({pct_conf:.1f}%)"},
                    {"label": "Réservations Annulées (Risque)", "value": f"{n_annulees} ({pct_ann:.1f}%)"},
                ]
                table_pred_pdf = [["Indicateur", "Valeur"]] + [[s["label"], s["value"]] for s in summary_pred_pdf]

                pdf_pred_bytes = generate_pdf_report(
                    title="Rapport Prédictif des Annulations",
                    subtitle="Booking AI — Module d'Anticipation Machine Learning",
                    summary_data=summary_pred_pdf,
                    table_data=table_pred_pdf,
                )

                st.download_button(
                    "📥 Télécharger le rapport prédictif (PDF)",
                    data=pdf_pred_bytes,
                    file_name=f"rapport_prediction_booking_ai_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                )
