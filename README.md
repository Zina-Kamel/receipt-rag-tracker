# Receipt Tracker and Financial Advisor App

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0F0F0F?style=flat&logo=mlflow&logoColor=white)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-00B4AB?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABh0lEQVRYR+2XvU4CQRCGv3dP+GR5CM9AjlDQpClZVNoXYCLv0gXcAy8AHoE6hCKRBaC2I1qBlJxB8sWZykVYhB7ptvNz0hG8pWdgLe2JmrPkyMxkqWQBa4BRWJptACnkAC7AFkJSfXZg7Ajm8Bq4BwgEXxD9mn+6z/YCz4+M+VIKrDngBzGg9DJXwMioQLeDRAi4L0O1fB8x7oGqag4mvQe2wWyV1ayE/3gXk1sTq7T1BRsB4S1rdxMyc0MW+3h8u05c1GmVGJe0BB1hS3kIevYPgwWDl8Bw0wWADb3M/gN4fsyx4V7f2GbrN+dj/AOaUgTk8+0PQeZ3q9V5c8G8oJTYKQAAAAASUVORK5CYII=)
![PDFPlumber](https://img.shields.io/badge/PDFPlumber-6E44FF?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABp0lEQVRYR+2Xz0sUURTHf2fBQI3kAErgIgsLoSx0KkIUEIoV9rUyFhJ6CUCUq+V4Z6UJOhJDco9uIrhH2+5zvL93ZyL5DEz3z+fHfvPd8kH0gRYRzFiP3Bqglr1Q8t0ArEDn4IGYPuAUmAGW0BtwC3k0VAv2ILpYgkQCp+AjWAsRgD1MVrYO0Y8IvkS8VwI3gK7AK6AO6DGg8wH2cKn9gPGiH7H0JuwD0MLJzCnjWgVwM34KYhCzwFYGr0u2kK7xEnCeoAxyDc4BbwIuA08G4pUX1KuWq6pD1IBd4QjUNwXr6FsgFviM+3RlyAQrxlZsrrOyDNzprl8PWwKvJpXAfR96+kh4KX8gZbsygyKP5D3I3+gE9z4e7AX+wU+dxnlPj5nB/lnm7D3nyc7nknP2Azyb3GpejG3tAAAAAElFTkSuQmCC)
![LoRA](https://img.shields.io/badge/LoRA-FF9900?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABHUlEQVRYR+2XQQrCQBRFz8t4kQCoWgRQCr6BwChgC0QAUINBBCl8IbRr0RZ0u2yOBZ2abmZOf3m7N79uBDOqzUkaYCeUoIO6DAxwB6qAZqJ7sAOXHzH+IOJXgCEAd8F2gENfY2lXcCbSBs1+Ek7wBLiKfEj1wK7gPK3r3zW8Ax4KFoK7gJTwLPANuBvIBxKHiC3Ak+BPYNK8q4H4EXWkJ/xe81V7cdkGe6I9MH19p+fBb+AA4g0AA5bz0QAAAAASUVORK5CYII=)
![Hugging Face](https://img.shields.io/badge/HuggingFace-F9900F?style=flat&logo=huggingface&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-2DD2FF?style=flat&logo=langchain&logoColor=white)
![LangSmith](https://img.shields.io/badge/LangSmith-FF3C00?style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABGklEQVRYR+2XwQ3CQBCFv3ZgADqAD6AA6gA9gA+gAeoAPYAOkAHoA/i8NiIYLKm3Z75i+rrICUSZiBm2hBA+A0vZAwZ4ANgPbiAwWYB1WIzTn2lHVpI0FmSTa4xyoQ3gAqAv4BrgA7wJeAK0wYoUZvA0zogPVgD+MCPtMZVtXrJg7yJ0i56fZwA3jPGH6pTuwk5rwdpXJb+JXs3uLZ4F24Bzx3Z7eKFcR8AAAAASUVORK5CYII=)
![FAISS](https://img.shields.io/badge/FAISS-0080FF?style=flat&logo=faiss&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-FF9900?style=flat&logo=nvidia&logoColor=white)

This RAG-powered web application simplifies receipt management and financial insights retreival. Users upload their receipts, automatically extract and store the information, and ask questions to quickly retrieve details or summaries from all stored receipts. Additionally, the built-in AI financial advisor provides personalized, actionable recommendations based on the spending patterns and and relevant scraped tips. 

## Features

- **OCR Extraction**: Automatically extract text from images and PDFs using PaddleOCR.
- **LLM-Based Parsing**: LoRA Fine-tuned LLM to convert raw OCR text into a structured JSON schema receipts.
- **FAISS Storage**: Store and index receipts for retrieval-augmented generation queries.
- **Spending Analysis**: Generate summaries, top vendors, monthly spend, and other spending statistics.
- **Interactive Queries**: Ask both aggregate questions (totals, counts, averages) and RAG queries (details from specific receipts).
- **PDF Reporting**: Export detailed financial reports.
- **AI Financial Advisor**: Receive actionable tips based on your receipts and spending habits.


## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Zina-Kamel/receipt-rag-tracker.git
cd receipt-rag-tracker
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

In .scretes.toml file, add your API Keys

```bash
GEMINI_API_KEY=
LANGSMITH_API_KEY=
OPENAI_API_KEY=
```
### 4. Run the Streamlit app

```bash
streamlit run app.py
```









