"""Booking AI — Streamlit Application in English with sidebar navigation and PDF reports."""

import csv
from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Import ReportLab for PDF report generation
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
# Utility Functions: Data Loading & Cleaning
# ---------------------------------------------------------------------------

@st.cache_resource
def load_trained_pipeline(model_path: Path):
    """Loads the joblib model with error handling for version mismatch."""
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.warning(f"⚠️ Model file incompatibility: {e}")
            return None
    return None


def load_data(uploaded_file) -> pd.DataFrame:
    """Loads a CSV/Excel file with basic validation (file size, extension)."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Extension not allowed: {suffix}")
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"File too large (> {MAX_FILE_SIZE_MB} MB)")

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

    raise ValueError(f"Unable to read this CSV file (unrecognized encoding/delimiter): {last_error}")


def iqr_bounds(series: pd.Series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def sanitize_for_excel(value):
    """Prevents formula injection in exported Excel files."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@"):
        return "'" + value
    return value


# ---------------------------------------------------------------------------
# Utility Functions: PDF Report Generation
# ---------------------------------------------------------------------------

def generate_pdf_report(title: str, subtitle: str, summary_data: list, table_data: list) -> bytes:
    """Generates a structured PDF report using ReportLab."""
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
        Paragraph(f"{subtitle} — Generated on {datetime.now().strftime('%d/%m/%Y at %H:%M')}", sub_style),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3B82F6"), spaceAfter=15),
    ]

    # Executive Summary Section
    story.append(Paragraph("<b>Key Indicators Summary:</b>", section_style))
    for item in summary_data:
        story.append(Paragraph(f"• <b>{item['label']}:</b> {item['value']}", body_style))

    story.append(Spacer(1, 15))

    # Data Table
    if table_data:
        story.append(Paragraph("<b>Synthetic Data Overview:</b>", section_style))
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
# Sidebar Navigation
# ---------------------------------------------------------------------------

st.sidebar.title("🏨 Booking AI")
st.sidebar.markdown("---")

menu_option = st.sidebar.radio(
    " Main Navigation",
    [
        "📌 Overview & Presentation",
        "📊 Data Exploration & Dashboards",
        "🤖 Cancellation Prediction",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Booking AI v2.0 • Decision Platform")


# ===========================================================================
# Section 1: Overview & Presentation
# ===========================================================================
if menu_option == "📌 Overview & Presentation":
    st.title("📌 Overview & Presentation")
    st.subheader("From raw data to strategic tourism decisions.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
            Welcome to **Booking AI**, your business intelligence solution for the hospitality and tourism sector.

            ### 🎯 Platform Objectives:
            1. **Monitoring & KPIs**: Real-time tracking of essential hotel performance metrics.
            2. **Exploratory Data Analysis**: Data quality audit, descriptive statistics, and outlier detection.
            3. **AI Cancellation Prediction**: Forecasting booking cancellations using Machine Learning to optimize Yield Management.
            """
        )
    with col2:
        st.info(
            """
            💡 **Quick Guide:**
            - Go to **Data Exploration & Dashboards** to upload and inspect your booking datasets.
            - Use the **Cancellation Prediction** tab to predict risks on upcoming customer reservations.
            """
        )
# ===========================================================================
# Section 2: Data Exploration and Dashboards
# ===========================================================================
elif menu_option == "📊 Data Exploration & Dashboards":
    st.header("📊 Data Exploration & Dashboards")

    file2 = st.file_uploader("Upload a file (Excel or CSV)", type=["csv", "xlsx"], key="explore")

    if file2:
        try:
            df = load_data(file2)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        if DATE_COL in df.columns:
            df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(15), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", len(df))
        c2.metric("Variables", df.shape[1])
        c3.metric("Duplicates", int(df.duplicated().sum()))
        c4.metric("Missing Values", int(df.isna().sum().sum()))

        st.markdown("---")
        st.subheader("📌 Categorized Descriptive Statistics")

        num_df = df.select_dtypes(include=np.number)
        cat_df = df.select_dtypes(include=["object", "category"])
        date_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()

        tab_num, tab_cat, tab_date = st.tabs(
            ["🔢 Numerical Variables", "🔤 Categorical Variables", "📅 Date Variables"]
        )

        with tab_num:
            if not num_df.empty:
                st.dataframe(num_df.describe().T, use_container_width=True)
            else:
                st.info("No numerical variables detected.")

        with tab_cat:
            if not cat_df.empty:
                st.dataframe(cat_df.describe(include="all").T, use_container_width=True)
            else:
                st.info("No categorical variables detected.")

        with tab_date:
            if date_cols:
                date_stats = []
                for col in date_cols:
                    s = df[col].dropna()
                    date_stats.append(
                        {
                            "Variable": col,
                            "Min Date": s.min().strftime("%Y-%m-%d") if not s.empty else "N/A",
                            "Max Date": s.max().strftime("%Y-%m-%d") if not s.empty else "N/A",
                            "Range (Days)": (s.max() - s.min()).days if not s.empty else 0,
                            "Missing Values": int(df[col].isna().sum()),
                        }
                    )
                st.dataframe(pd.DataFrame(date_stats), use_container_width=True)
            else:
                st.info("No date columns identified.")

        # Outliers & Correlation Matrix
        st.markdown("---")
        col_out, col_corr = st.columns(2)

        with col_out:
            with st.container(border=True):
                st.subheader("Outliers (IQR Method)")
                num_cols = num_df.columns.tolist()
                rows = []
                for col in num_cols:
                    lower, upper = iqr_bounds(df[col])
                    mask = (df[col] < lower) | (df[col] > upper)
                    rows.append(
                        {
                            "variable": col,
                            "outliers_count": int(mask.sum()),
                            "lower_bound": round(lower, 2),
                            "upper_bound": round(upper, 2),
                        }
                    )
                outlier_df = pd.DataFrame(rows).sort_values("outliers_count", ascending=False)
                st.dataframe(outlier_df, height=220, use_container_width=True)

        with col_corr:
            with st.container(border=True):
                st.subheader("Correlation Matrix")
                if len(num_cols) > 1:
                    corr = df[num_cols].corr()
                    fig_corr = px.imshow(
                        corr,
                        text_auto=".2f",
                        color_continuous_scale="Blues",
                        aspect="auto",
                        title="Linear Correlations",
                    )
                    fig_corr.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Insufficient numerical variables to display a correlation matrix.")

        # Dashboards and KPIs
        st.markdown("---")
        st.subheader("📈 Decision Dashboards & KPIs")

        # 4-Filter System
        st.markdown("**🎛️ Combined Filters:**")
        fk1, fk2, fk3, fk4 = st.columns(4)

        date_range = None
        if DATE_COL in df.columns and df[DATE_COL].notna().any():
            min_d, max_d = df[DATE_COL].min(), df[DATE_COL].max()
            date_range = fk1.date_input("1. Filter by Date", (min_d, max_d))

        hotel_filter = "All"
        if HOTEL_COL in df.columns:
            hotel_opts = ["All"] + sorted(df[HOTEL_COL].dropna().unique().tolist())
            hotel_filter = fk2.selectbox("2. Hotel Type", hotel_opts)

        loc_filter = "All"
        if LOCATION_COL in df.columns:
            loc_opts = ["All"] + sorted(df[LOCATION_COL].dropna().unique().tolist())
            loc_filter = fk3.selectbox("3. Location / City", loc_opts)

        chan_filter = "All"
        if CHANNEL_COL in df.columns:
            chan_opts = ["All"] + sorted(df[CHANNEL_COL].dropna().unique().tolist())
            chan_filter = fk4.selectbox("4. Distribution Channel", chan_opts)

        # Apply Filters
        dff = df.copy()
        if date_range and len(date_range) == 2 and DATE_COL in dff.columns:
            dff = dff[
                (dff[DATE_COL] >= pd.to_datetime(date_range[0]))
                & (dff[DATE_COL] <= pd.to_datetime(date_range[1]))
            ]
        if hotel_filter != "All" and HOTEL_COL in dff.columns:
            dff = dff[dff[HOTEL_COL] == hotel_filter]
        if loc_filter != "All" and LOCATION_COL in dff.columns:
            dff = dff[dff[LOCATION_COL] == loc_filter]
        if chan_filter != "All" and CHANNEL_COL in dff.columns:
            dff = dff[dff[CHANNEL_COL] == chan_filter]

        # Enclosed Visualizations
        g1, g2 = st.columns(2)
        if TARGET_COL in dff.columns:
            status_map = {0: "Confirmed", 1: "Canceled"}
            dff["_statut"] = dff[TARGET_COL].map(status_map).fillna(dff[TARGET_COL].astype(str))

            with g1:
                with st.container(border=True):
                    fig_pie = px.pie(
                        dff,
                        names="_statut",
                        title="Cancellation Distribution (Filtered Data)",
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
                            title="Bookings by Hotel Type",
                        )
                        st.plotly_chart(fig_bar_h, use_container_width=True)
                    else:
                        st.info("'type_hotel' column missing.")

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
                            title="Bookings by City / Location",
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
                            title="Bookings by Distribution Channel",
                        )
                        st.plotly_chart(fig_chan, use_container_width=True)

        # PDF Exploration Report
        st.markdown("---")
        st.subheader("📄 Export Exploration Report")

        summary_pdf = [
            {"label": "Total Filtered Records", "value": str(len(dff))},
            {"label": "Analyzed Period", "value": f"{date_range[0]} to {date_range[1]}" if date_range else "All-Time"},
            {"label": "Hotel Filter", "value": hotel_filter},
            {"label": "City Filter", "value": loc_filter},
            {"label": "Channel Filter", "value": chan_filter},
        ]

        if TARGET_COL in dff.columns:
            nb_ann = int((dff[TARGET_COL] == 1).sum())
            taux_ann = round((nb_ann / len(dff) * 100), 2) if len(dff) > 0 else 0
            summary_pdf.append({"label": "Cancellation Rate", "value": f"{taux_ann}% ({nb_ann} cancellations)"})

        table_data_pdf = [["Metric", "Value"]] + [[s["label"], s["value"]] for s in summary_pdf]

        pdf_bytes = generate_pdf_report(
            title="Exploration & Dashboards Report",
            subtitle="Booking AI — Decision Analytics",
            summary_data=summary_pdf,
            table_data=table_data_pdf,
        )

        st.download_button(
            "📥 Download Exploration Report (PDF)",
            data=pdf_bytes,
            file_name=f"exploration_report_booking_ai_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
        )


# ===========================================================================
# Section 3: Cancellation Prediction
# ===========================================================================
elif menu_option == "🤖 Cancellation Prediction":
    st.header("🤖 Cancellation Prediction")
    st.caption("Upload a dataset without target labels to generate model predictions.")

    pipeline = load_trained_pipeline(MODEL_PATH)

    if pipeline is None:
        st.error(
            "Model file not found (`model/booking_ai_pipeline.joblib`). "
            "Please run `train_model.py` first."
        )
    else:
        file3 = st.file_uploader(
            "Upload file to predict (Excel or CSV)", type=["csv", "xlsx"], key="predict"
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
                if hasattr(pipeline, "predict_proba"):
                    probs = pipeline.predict_proba(X)[:, 1]
                else:
                    probs = None
                    preds_default = pipeline.predict(X)
            except Exception as e:
                st.error(
                    "Error during prediction — check if file columns match expected schema.\n\n"
                    f"Technical Details: {e}"
                )
                st.stop()

            # Dynamic Risk Threshold Tuning
            st.markdown("---")
            st.subheader("⚙️ Risk Classification Threshold Setting")

            threshold = st.slider(
                "Probability threshold to classify a booking as 'Canceled'",
                min_value=0.10,
                max_value=0.90,
                value=0.50,
                step=0.05,
                help="Bookings with cancellation probabilities exceeding this threshold will be flagged as 'Canceled'."
            )

            if probs is not None:
                preds = np.where(probs >= threshold, 1, 0)
            else:
                preds = preds_default

            result = pd.DataFrame(
                {
                    DATE_COL: dates.values,
                    "predicted_status": np.where(preds == 1, "Canceled", "Confirmed"),
                }
            )
            if probs is not None:
                result["cancellation_probability_%"] = (probs * 100).round(2)

            result = result.map(sanitize_for_excel) if hasattr(result, "map") else result.applymap(sanitize_for_excel)

            n_confirmees = int((preds == 0).sum())
            n_annulees = int((preds == 1).sum())
            total_preds = len(preds)

            st.markdown("---")
            st.subheader("📊 Predicted Bookings Summary")

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Bookings Predicted", total_preds)
            m2.metric("Predicted Confirmed 🟢", n_confirmees)
            m3.metric("Predicted Canceled 🔴", n_annulees)

            pred_summary = pd.DataFrame(
                {
                    "Status": ["Confirmed", "Canceled"],
                    "Count": [n_confirmees, n_annulees]
                }
            )

            # Enclosed Visualizations
            cp1, cp2 = st.columns(2)
            with cp1:
                with st.container(border=True):
                    if total_preds > 0:
                        fig_pred_pie = px.pie(
                            pred_summary,
                            names="Status",
                            values="Count",
                            title="Predicted Cancellations Breakdown",
                            color="Status",
                            color_discrete_map={"Confirmed": "#10B981", "Canceled": "#EF4444"},
                            hole=0.3
                        )
                        st.plotly_chart(fig_pred_pie, use_container_width=True)
                    else:
                        st.info("No data available to display.")

            with cp2:
                with st.container(border=True):
                    if probs is not None:
                        fig_hist = px.histogram(
                            result,
                            x="cancellation_probability_%",
                            nbins=20,
                            title="Cancellation Probability Distribution (%)",
                            color_discrete_sequence=["#3B82F6"],
                        )
                        fig_hist.add_vline(
                            x=threshold * 100, 
                            line_dash="dash", 
                            line_color="red", 
                            annotation_text=f"Threshold ({int(threshold*100)}%)"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    else:
                        st.info("Probabilities not supported by the loaded model.")

            st.subheader("Predicted Results Table")
            st.dataframe(result, use_container_width=True)

            # Download Options (Excel + PDF)
            col_d1, col_d2 = st.columns(2)

            with col_d1:
                buffer_excel = BytesIO()
                with pd.ExcelWriter(buffer_excel, engine="openpyxl") as writer:
                    result.to_excel(writer, index=False, sheet_name="predictions")
                st.download_button(
                    "📥 Download Predictions (Excel)",
                    data=buffer_excel.getvalue(),
                    file_name="predictions_booking_ai.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            with col_d2:
                pct_conf = (n_confirmees / total_preds * 100) if total_preds > 0 else 0
                pct_ann = (n_annulees / total_preds * 100) if total_preds > 0 else 0

                summary_pred_pdf = [
                    {"label": "Total Analyzed Volume", "value": str(total_preds)},
                    {"label": "Applied Risk Threshold", "value": f"{int(threshold * 100)}%"},
                    {"label": "Confirmed Bookings", "value": f"{n_confirmees} ({pct_conf:.1f}%)"},
                    {"label": "Canceled Bookings (At Risk)", "value": f"{n_annulees} ({pct_ann:.1f}%)"},
                ]
                table_pred_pdf = [["Indicator", "Value"]] + [[s["label"], s["value"]] for s in summary_pred_pdf]

                pdf_pred_bytes = generate_pdf_report(
                    title="Cancellation Prediction Report",
                    subtitle="Booking AI — Machine Learning Risk Forecasting",
                    summary_data=summary_pred_pdf,
                    table_data=table_pred_pdf,
                )

                st.download_button(
                    "📥 Download Prediction Report (PDF)",
                    data=pdf_pred_bytes,
                    file_name=f"prediction_report_booking_ai_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                )
