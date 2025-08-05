import streamlit as st
from PIL import Image
from core import ocr, extract, embed, store, anomaly
from core.langchain_rag import get_rag_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import plotly.express as px
from io import BytesIO
from core.ocr import pdf_to_images, run_paddleocr
from datetime import datetime
from fpdf import FPDF  

if "faiss_store" not in st.session_state:
    st.session_state.faiss_store = store.FaissStore()
    try:
        st.session_state.faiss_store.load()
    except FileNotFoundError:
        pass

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = get_rag_chain()

faiss_store = st.session_state.faiss_store
rag_chain = st.session_state.rag_chain

st.title("Receipt Insight MVP")

st.header("Spending Dashboard")

metadata = faiss_store.metadata  

if metadata:
    df = pd.DataFrame(metadata)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['total'] = pd.to_numeric(df['total'], errors='coerce')
    df = df.dropna(subset=['date', 'total'])

    valid_dates = df['date'].dropna()
    if valid_dates.empty:
        st.info("No valid dates found in data. Showing all data without date filtering.")
        filtered = df.copy()
        start_date = None
        end_date = None
    else:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

        start_date, end_date = st.date_input(
            "Filter by Date Range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        filtered = df[
            (df['date'] >= pd.Timestamp(start_date)) &
            (df['date'] <= pd.Timestamp(end_date))
        ]


    categories = df['category'].dropna().unique().tolist() if 'category' in df.columns else []
    selected_categories = st.multiselect(
        "Filter by Categories:",
        options=categories,
        default=categories
    )

    if selected_categories:
        filtered = filtered[filtered['category'].isin(selected_categories)]

    if filtered.empty:
        st.info("No data for selected filters.")
    else:
        filtered['month'] = filtered['date'].dt.to_period('M').dt.to_timestamp()

        st.subheader("Monthly Spend by Category")
        if 'category' in filtered.columns and not filtered['category'].isnull().all():
            monthly = filtered.groupby(['month', 'category'])['total'].sum().reset_index()
            fig1 = px.bar(
                monthly,
                x='month', y='total', color='category',
                labels={'total': 'Total Spend', 'month': 'Month'},
                title='Monthly Spend by Category',
                barmode='stack'
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("No categories available for this data.")

        st.subheader("Top Vendors by Total Spend")
        vendor_totals = filtered.groupby('vendor')['total'].sum().sort_values(ascending=False).head(5).reset_index()
        fig2 = px.bar(
            vendor_totals,
            x='vendor', y='total',
            labels={'total': 'Total Spend', 'vendor': 'Vendor'},
            title='Top 5 Vendors'
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Total Spend Over Time")
        time_series = filtered.groupby('date')['total'].sum().reset_index()
        fig3 = px.line(
            time_series,
            x='date', y='total',
            labels={'total': 'Total Spend', 'date': 'Date'},
            title='Daily Total Spend Over Time'
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Spending Anomalies (Visual)")
        threshold = filtered['total'].mean() + 2 * filtered['total'].std()
        anomalies = filtered[filtered['total'] > threshold]
        if not anomalies.empty:
            st.warning(f"{len(anomalies)} anomalies detected (spending > {threshold:.2f})")
            st.dataframe(anomalies[['vendor', 'date', 'total', 'category']])
        else:
            st.success("No anomalies detected.")

        st.subheader("Export Insights")
        csv = filtered.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "receipt_data_filtered.csv", "text/csv")

        st.subheader("Export PDF Report")

        def generate_pdf(dataframe):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Receipt Spending Report", ln=True, align="C")

            pdf.set_font("Arial", size=12)
            pdf.ln(10)
            if start_date and end_date:
                pdf.cell(0, 10, f"Date Range: {start_date} to {end_date}", ln=True)
            else:
                pdf.cell(0, 10, "Date Range: All Data", ln=True)

            pdf.ln(10)
            pdf.cell(0, 10, "Top 5 Vendors by Spend:", ln=True)
            pdf.set_font("Arial", size=10)
            top_vendors = dataframe.groupby('vendor')['total'].sum().sort_values(ascending=False).head(5)
            for vendor, total in top_vendors.items():
                pdf.cell(0, 8, f"{vendor}: ${total:.2f}", ln=True)

            pdf.ln(10)
            total_spent = dataframe['total'].sum()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Total Spend in Period: ${total_spent:.2f}", ln=True)

            anomaly_count = anomalies.shape[0]
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, f"Anomalies Detected: {anomaly_count}", ln=True)

            return pdf.output(dest='S').encode('latin1')

        pdf_bytes = generate_pdf(filtered)
        st.download_button("Download PDF Report", pdf_bytes, "spending_report.pdf", "application/pdf")

else:
    st.info("No receipt data available yet. Upload receipts to see analytics.")


st.header("Upload New Receipts")

uploaded = st.file_uploader("Upload receipt image or PDF", type=["jpg", "jpeg", "png", "pdf"], key="upload_new")
query = st.text_input("Ask a question about your receipts", key="query_new")

if uploaded:
    if uploaded.type == "application/pdf":
        st.info("Processing PDF receipt pages...")
        images = pdf_to_images(uploaded)
    else:
        images = [Image.open(uploaded)]

    for i, image in enumerate(images):
        st.image(image, caption=f"Page {i+1}" if len(images) > 1 else "Uploaded Receipt")

        text = run_paddleocr(image)
        
        st.text_area("OCR Output", text, height=200)

        fields = extract.extract_fields(text)
        st.json(fields)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        for chunk in chunks:
            embedding = embed.embed_text(chunk)
            faiss_store.add(embedding, chunk, fields)

    faiss_store.save()

    st.session_state.rag_chain = get_rag_chain()
    rag_chain = st.session_state.rag_chain

    query_emb = embed.embed_text("test query")
    results = faiss_store.search(query_emb, k=5)
    st.write(f"Sample search results count: {len(results)}")

    is_anomaly = anomaly.detect_anomaly(fields, faiss_store.metadata)
    if is_anomaly:
        st.warning(f"Anomaly detected: This receipt total ({fields['total']}) is significantly higher than usual for vendor {fields['vendor']}.")

if query:
    if not faiss_store.texts:
        st.info("Upload some receipts first!")
    else:
        with st.spinner("Generating answer..."):
            result = rag_chain.invoke({"query": query})
            st.markdown(f"**Answer:** {result['result']}")
