import os
import ast
import json
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from fpdf import FPDF
from langchain.embeddings import HuggingFaceEmbeddings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from core import store
from core.langchain_rag import get_rag_chain
from core.ocr import create_ocr_engine, pdf_to_images, run_paddleocr
from langchain_google_genai import ChatGoogleGenerativeAI
import plotly.express as px

from config import settings

PYTORCH_CUDA_ALLOC_CONF = settings.PYTORCH_CUDA_ALLOC_CONF
GEMINI_API_KEY = settings.GEMINI_API_KEY
OFFLOAD_DIR = settings.OFFLOAD_DIR
FINETUNED_MODEL_DIR = settings.FINETUNED_MODEL_DIR
BASE_MODEL_NAME = settings.BASE_MODEL_NAME
EMBEDDING_MODEL_NAME = settings.EMBEDDING_MODEL_NAME
PDF_TITLE = settings.PDF_TITLE
FAISS_RECEIPTS_STORE_NAME = settings.FAISS_RECEIPTS_STORE_NAME
FAISS_TIPS_STORE_NAME = settings.FAISS_TIPS_STORE_NAME
APP_TITLE = settings.APP_NAME

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = PYTORCH_CUDA_ALLOC_CONF
os.makedirs(OFFLOAD_DIR, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"}
)

@st.cache_resource(show_spinner=False)
def load_finetuned_model(model_dir=FINETUNED_MODEL_DIR):
    offload_dir = OFFLOAD_DIR
    os.makedirs(offload_dir, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    base_model_name = BASE_MODEL_NAME

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
        offload_folder=offload_dir,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base_model, model_dir, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model, device

tokenizer, model, device = load_finetuned_model()

def parse_ocr_words(text):
    try:
        outer = json.loads(text)
        ocr_str = outer.get("ocr_words", "[]")
        ocr_list = ast.literal_eval(ocr_str)
        return ocr_list if isinstance(ocr_list, list) else [str(ocr_list)]
    except Exception:
        return [text]

def generate_json(text, max_retries=5, allowed_categories=""):
    """
    Generate structured JSON from OCR text using the fine-tuned model.
    Retries up to max_retries times if output is invalid.
    """
    ocr_words_list = parse_ocr_words(text)
    ocr_words_str = "\n".join(ocr_words_list)
 
    base_prompt = f"""Extract information from the following receipt OCR text.

    OCR Text:
    {ocr_words_str}

    Required JSON format:
    {{
    "store": "string",
    "date": "string", 
    "items": [
        {{"name": "string", "price": 0.0}}
    ],
    "category": "string (must be one of: {allowed_categories})",
    "total": 0.0
    }}

    JSON:"""

    for attempt in range(max_retries):
        inputs = tokenizer(base_prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        torch.cuda.empty_cache()
        
        try:
            with torch.no_grad():  
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.3,  
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1, 
                )
        except RuntimeError as e:
            if device.type == "cuda":
                inputs_cpu = {k: v.cpu() for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs_cpu,
                        max_new_tokens=500,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.9,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                    )
            else:
                raise e

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            last_open_brace = generated.rfind('JSON:')+5  
            
            last_close_brace = generated.rfind('}')
            
            if last_open_brace != -1 and last_close_brace != -1 and last_close_brace > last_open_brace:
                json_str = generated[last_open_brace:last_close_brace + 1]
                parsed = json.loads(json_str)
                required_fields = ["store", "date", "total", "items", "category"]
                if all(field in parsed for field in required_fields) and isinstance(parsed["items"], list):
                    return parsed
                    
        except Exception:
            pass
    
    return {
        "error": "Failed to generate valid JSON after all retries", 
        "raw_output": generated if 'generated' in locals() else "No output"
    }


def total_spent_by_category(df, start_date=None, end_date=None):
    filtered = df.copy()
    if start_date:
        filtered = filtered[filtered['date'] >= pd.Timestamp(start_date)]
    if end_date:
        filtered = filtered[filtered['date'] <= pd.Timestamp(end_date)]
    return filtered.groupby('category')['total'].sum().sort_values(ascending=False)

def average_spent(df, start_date=None, end_date=None):
    filtered = df.copy()
    if start_date:
        filtered = filtered[filtered['date'] >= pd.Timestamp(start_date)]
    if end_date:
        filtered = filtered[filtered['date'] <= pd.Timestamp(end_date)]
    return filtered['total'].mean()

def count_receipts_in_period(df, start_date=None, end_date=None):
    filtered = df.copy()
    if start_date:
        filtered = filtered[filtered['date'] >= pd.Timestamp(start_date)]
    if end_date:
        filtered = filtered[filtered['date'] <= pd.Timestamp(end_date)]
    return len(filtered)

def total_spent_in_period(df, start_date=None, end_date=None):
    filtered = df.copy()
    if start_date:
        filtered = filtered[filtered['date'] >= pd.Timestamp(start_date)]
    if end_date:
        filtered = filtered[filtered['date'] <= pd.Timestamp(end_date)]
    return filtered['total'].sum()

def top_vendors_by_spend(df, top_n=5, start_date=None, end_date=None):
    filtered = df.copy()
    if start_date:
        filtered = filtered[filtered['date'] >= pd.Timestamp(start_date)]
    if end_date:
        filtered = filtered[filtered['date'] <= pd.Timestamp(end_date)]
    return filtered.groupby('store')['total'].sum().sort_values(ascending=False).head(top_n)

def classify_query_type(query: str, llm, threshold=0.5):
    prompt = f"""
    You are a helpful assistant that classifies a user's query into two categories: "aggregate" or "rag".
    "aggregate" means statistics, totals, counts, averages, or comparisons across multiple receipts.
    "rag" means details from specific receipts or textual information retrieval.
    Classify ONLY, output exactly one word: aggregate or rag.
    Query: "{query}"
    Answer:
    """
    response = llm.invoke(prompt)
    classification = response.content.strip().lower()
    return classification if classification in ("aggregate","rag") else "rag"

def generate_pdf(dataframe, start_date=None, end_date=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, settings.get("PDF_TITLE"), ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    if start_date and end_date:
        pdf.cell(0, 10, f"Date Range: {start_date} to {end_date}", ln=True)
    else:
        pdf.cell(0, 10, "Date Range: All Data", ln=True)
    pdf.ln(10)
    pdf.cell(0, 10, "Top 5 Vendors by Spend:", ln=True)
    pdf.set_font("Arial", size=10)
    top_vendors = dataframe.groupby('store')['total'].sum().sort_values(ascending=False).head(5)
    for vendor, total in top_vendors.items():
        pdf.cell(0, 8, f"{vendor}: ${total:.2f}", ln=True)
    pdf.ln(10)
    total_spent = dataframe['total'].sum()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Total Spend in Period: ${total_spent:.2f}", ln=True)
    return pdf.output(dest='S').encode('latin1')

def init_faiss_store(store_name, example_embedding=None):
    dim = example_embedding.shape[0] if example_embedding is not None else 384
    fs = store.FaissStore(dim=dim, store_name=store_name)
    try: fs.load()
    except FileNotFoundError: pass
    return fs

if "faiss_store" not in st.session_state:
    st.session_state.faiss_store = init_faiss_store(FAISS_RECEIPTS_STORE_NAME)
if "faiss_tips_store" not in st.session_state:
    st.session_state.faiss_tips_store = init_faiss_store(FAISS_TIPS_STORE_NAME)
if "receipt_rag_chain" not in st.session_state:
    st.session_state.receipt_rag_chain = get_rag_chain(store_name=FAISS_RECEIPTS_STORE_NAME)
if "tips_rag_chain" not in st.session_state:
    st.session_state.tips_rag_chain = get_rag_chain(store_name=FAISS_TIPS_STORE_NAME)

faiss_store = st.session_state.faiss_store
faiss_tips_store = st.session_state.faiss_tips_store
receipts_rag_chain = st.session_state.receipt_rag_chain

st.title(APP_TITLE)
tabs = st.tabs(["Receipts", "Analysis", "Financial Advisor"])

def normalize_receipt_metadata(receipt):
    """
    Normalize receipt metadata for FAISS storage:
    - Lowercase and strip store names
    - Convert dates to ISO format (YYYY-MM-DD)
    - Convert total and item prices to float
    """
    normalized = receipt.copy()

    if 'store' in normalized and isinstance(normalized['store'], str):
        normalized['store'] = normalized['store'].strip().lower()
    
    if 'date' in normalized:
        try:
            dt = pd.to_datetime(normalized['date'], errors='coerce')
            normalized['date'] = dt.strftime("%Y-%m-%d") if not pd.isna(dt) else normalized['date']
        except Exception:
            pass

    if 'total' in normalized:
        try:
            normalized['total'] = float(normalized['total'])
        except:
            pass

    if 'items' in normalized and isinstance(normalized['items'], list):
        for item in normalized['items']:
            if 'price' in item:
                try:
                    item['price'] = float(item['price'])
                except:
                    pass
            if 'name' in item and isinstance(item['name'], str):
                item['name'] = item['name'].strip()
                
    if 'category' in normalized and isinstance(normalized['category'], str):
        normalized['category'] = normalized['category'].strip().lower()

    return normalized


with tabs[0]:
    allowed_categories_input = st.text_input(
    "Enter the categories you want the receipts to be classified to (comma-separated):",
    value="food,transport,entertainment,shopping,utilities,grocery"
    )
    allowed_categories = [c.strip().lower() for c in allowed_categories_input.split(",") if c.strip()]

    st.header("Upload New Receipts")
    uploaded = st.file_uploader("Upload receipt image or PDF", type=["jpg","jpeg","png","pdf"])
    query = st.text_input("Ask a question about your receipts:")
    print("query:", query)

    if uploaded:
        images = pdf_to_images(uploaded) if uploaded.type=="application/pdf" else [Image.open(uploaded)]
        for i, image in enumerate(images):
            st.image(image, caption=f"Page {i+1}" if len(images)>1 else "Uploaded Receipt")
            ocr_engine = create_ocr_engine()
            text = run_paddleocr(image, ocr_engine)
            st.text_area("OCR Output", text, height=200)
            llm_json = generate_json(text, allowed_categories=", ".join(allowed_categories))
            normalized_json = normalize_receipt_metadata(llm_json)
            st.json(llm_json)
            embedding = embedding_model.embed_documents([json.dumps(normalized_json, ensure_ascii=False)])[0]
            embedding = np.array(embedding, dtype=np.float32)
            faiss_store.add(embedding, json.dumps(normalized_json, ensure_ascii=False), normalized_json)

        faiss_store.save()
        st.success("Receipts processed successfully!")
        st.session_state.receipts_rag_chain = get_rag_chain(store_name=FAISS_RECEIPTS_STORE_NAME)
        receipts_rag_chain = st.session_state.receipts_rag_chain

    if query:
            print(f"User query: {query}")
            if not faiss_store.texts:
                st.info("Upload some receipts first!")
            else:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash-latest",
                    google_api_key=settings.GEMINI_API_KEY
                )
                query_type = classify_query_type(query, llm)
                df = pd.DataFrame(faiss_store.metadata)
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['total'] = pd.to_numeric(df['total'], errors='coerce')
                df = df.dropna(subset=['date','total'])

                with st.spinner("Generating answer..."):
                    if query_type == "rag":
                        print(f"RAG query detected: {query}")
                        result = receipts_rag_chain.invoke({"query": query})
                        st.markdown(f"**Answer:** {result['result']}")
                    else:  
                        answer_text = ""
                        q_lower = query.lower()
                        if "total by category" in q_lower or "spend by category" in q_lower:
                            totals = total_spent_by_category(df)
                            answer_text = "**Total spent by category:**\n" + "\n".join(
                                [f"{cat}: ${val:.2f}" for cat, val in totals.items()]
                            )
                        elif "average" in q_lower or "mean" in q_lower:
                            avg = average_spent(df)
                            answer_text = f"**Average spend per receipt:** ${avg:.2f}"
                        elif "count" in q_lower or "number of receipts" in q_lower:
                            count = count_receipts_in_period(df)
                            answer_text = f"**Number of receipts:** {count}"
                        elif "top vendors" in q_lower or "highest spending stores" in q_lower:
                            top = top_vendors_by_spend(df)
                            answer_text = "**Top vendors by spend:**\n" + "\n".join(
                                [f"{store}: ${total:.2f}" for store, total in top.items()]
                            )
                        else:
                            st.markdown("**Aggregate query not recognized.**")
                        
                        st.markdown(answer_text)


with tabs[1]:
    st.header("Spending Analysis")
    metadata = faiss_store.metadata
    if metadata:
        df = pd.DataFrame(metadata)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['total'] = pd.to_numeric(df['total'], errors='coerce')
        df = df.dropna(subset=['date','total'])
        min_date, max_date = df['date'].min(), df['date'].max()
        date_range = st.date_input(
            "Select date range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if isinstance(date_range, tuple) or isinstance(date_range, list):
            if len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = date_range[0]
                end_date = None
        else:
            start_date = date_range
            end_date = None

        if end_date is not None:
            st.write(f"Filtering from {start_date} to {end_date}")
        else:
            st.write(f"Start date selected: {start_date}, waiting for end date...")
        categories = df['category'].dropna().unique().tolist()
        selected_categories = st.multiselect(
            "Filter by category:",
            options=categories,
            default=categories
        )

        start_date_dt = pd.to_datetime(start_date)

        if end_date is not None:
            end_date_dt = pd.to_datetime(end_date)
            df = df[(df['date'] >= start_date_dt) & (df['date'] <= end_date_dt)]
        else:
            df = df[df['date'] >= start_date_dt]

        df = df[df['category'].isin(selected_categories)]
        
        df_display = df.copy()
        df_display['items'] = df_display['items'].apply(lambda x: json.dumps(x) if isinstance(x, list) else str(x))
        st.dataframe(df_display)


        if 'category' in df.columns:
            df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
            monthly = df.groupby(['month','category'])['total'].sum().reset_index()
            fig = px.bar(monthly, x='month', y='total', color='category', barmode='stack', title="Monthly Spend by Category")
            st.plotly_chart(fig, use_container_width=True)

        vendor_totals = df.groupby('store')['total'].sum().sort_values(ascending=False).head(5).reset_index()
        fig = px.bar(vendor_totals, x='store', y='total', title="Top 5 Vendors")
        st.plotly_chart(fig, use_container_width=True)

        daily = df.groupby('date')['total'].sum().reset_index()
        fig = px.line(daily, x='date', y='total', title="Daily Total Spend")
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Export PDF Report"):
            pdf_bytes = generate_pdf(df)
            st.download_button("Download PDF", pdf_bytes, "spending_report.pdf", "application/pdf")
    else:
        st.info("No receipts data to analyze yet.")

with tabs[2]:
    st.header("AI Financial Advisor")
    # if st.button("Fetch & Add New Tips"):
    #     tips_data = fetch_all_sources()
    #     for tip, source in tips_data:
    #         metadata = {"type": "tip", "content": tip, "source": source}
    #         embedding = embedding_model.embed_documents([tip])[0]
    #         embedding = np.array(embedding, dtype=np.float32)
    #         faiss_tips_store.add(embedding, tip, metadata)


    #     faiss_tips_store.save()
    #     st.success(f"Added {len(tips_data)} tips.")

    mode = st.radio("Mode", ["Chat with Advisor"])
    if mode == "Chat with Advisor":
        user_q = st.text_area("Ask the financial advisor:")
        if user_q:
            df = pd.DataFrame(faiss_store.metadata)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['total'] = pd.to_numeric(df['total'], errors='coerce')
            df = df.dropna(subset=['date','total'])
            total_spent = df['total'].sum() if not df.empty else 0
            top_cat = df.groupby('category')['total'].sum().sort_values(ascending=False).head(3).to_dict() if not df.empty else {}
            spending_summary = f"Total spent: ${total_spent:.2f}. Top categories: {top_cat}"
            tips_results = faiss_tips_store.search(np.array(embedding_model.embed_query(user_q), dtype=np.float32), k=15)

            tip_texts = [meta['content'] for _, meta in tips_results if meta.get("type") == "tip"]
            system_prompt = (
                "You are a practical personal finance coach. "
                "Use the retrieved TIPS as your factual base. "
                "Provide 3–6 actionable, concrete suggestions, tailored to the user's spending summary."
            )
            context = "\n".join(f"- {t}" for t in tip_texts)
            user_prompt = f"User question: {user_q}\n\nSpending summary: {spending_summary}\n\nRelevant tips:\n{context}"
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=settings.GEMINI_API_KEY)
            response = llm.invoke([{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}])
            st.markdown("### Advisor's Response:")
            st.write(response.content)
