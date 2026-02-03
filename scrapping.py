import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

load_dotenv()
API_KEY = os.getenv("Langchain")

st.set_page_config(page_title="Laptop AI Assistant", layout="wide")
st.title("Laptop AI Assistant with Web Scraping")

if not API_KEY:
    st.error("OpenAI API key not found. Please set LANGCHAIN_API in .env")
    st.stop()

@st.cache_data
def scrape_laptop_data():
    base_url = "https://webscraper.io/test-sites/e-commerce/static/computers/laptops?page="
    results = []

    for i in range(1, 21):
        response = requests.get(base_url + str(i)).text
        soup = BeautifulSoup(response, "html.parser")
        items = soup.find_all("div", class_="col-md-4 col-xl-4 col-lg-4")

        for item in items:
            results.append({
                "Product Name": item.find("a", class_="title").text.strip(),
                "Product Price": item.find("span").text,
                "Description": item.find("p", class_="description card-text").text,
                "Review": item.find("p", class_="review-count float-end").text.strip()
            })

    return pd.DataFrame(results)

df = scrape_laptop_data()

def row_to_text(row):
    return f"""
Product Name: {row['Product Name']}
Product Price: {row['Product Price']}
Description: {row['Description']}
Review: {row['Review']}
"""

documents = [
    Document(page_content=row_to_text(row))
    for _, row in df.iterrows()
]

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=API_KEY
)

vectorstore = FAISS.from_documents(documents, embeddings)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=API_KEY
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer ONLY using the context below.
If the answer is not present say:
"Not available in the provided data."

Context:
{context}

Question:
{question}
"""
)

chain = prompt | llm

def detect_question_type(query):
    q = query.lower()
    if any(x in q for x in ["list", "show", "all", "below", "under", "less than"]):
        return "FILTER"
    if any(x in q for x in ["best", "recommended", "suggest"]):
        return "RECOMMEND"
    return "FACT"

def extract_price(query):
    match = re.search(r"\$?(\d+)", query)
    return float(match.group(1)) if match else None

def extract_review(query):
    match = re.search(r"(\d+)\s*reviews?", query.lower())
    return int(match.group(1)) if match else 0

query = st.text_input("Ask a Question About Laptops:")

if query:
    qtype = detect_question_type(query)
    st.caption(f"Detected Question Type: {qtype}")

    if qtype == "FACT":
        docs = vectorstore.similarity_search(query, k=4)
        context_text = "\n".join(d.page_content for d in docs)
        response = chain.invoke({
            "context": context_text,
            "question": query
        })
        st.success(response.content)

    elif qtype == "FILTER":
        price_limit = extract_price(query)

        if price_limit is None:
            st.warning("Please specify a price.")
        else:
            df["Product Price"] = (
                df["Product Price"]
                .astype(str)
                .str.replace("$", "", regex=False)
                .astype(float)
            )

            filtered = df[df["Product Price"] < price_limit]

            if filtered.empty:
                st.warning("No laptops found under this price.")
            else:
                docs = filtered.apply(row_to_text, axis=1).tolist()
                context_text = "\n".join(docs[:10])
                response = chain.invoke({
                    "context": context_text,
                    "question": "List the laptop names with prices"
                })
                st.success(response.content)

    elif qtype == "RECOMMEND":
        price_limit = extract_price(query) or 1000
        min_reviews = extract_review(query)

        df["Product Price"] = (
            df["Product Price"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .astype(float)
        )

        df["Review"] = (
            df["Review"]
            .astype(str)
            .str.replace("reviews", "", regex=False)
            .astype(int)
        )

        filtered = df[
            (df["Product Price"] > price_limit) &
            (df["Review"] >= min_reviews)
        ]

        if filtered.empty:
            st.warning("No suitable laptops found.")
        else:
            docs = filtered.apply(row_to_text, axis=1).tolist()
            context_text = "\n".join(docs[:10])
            response = chain.invoke({
                "context": context_text,
                "question": "List the laptop names with prices and reviews"
            })
            st.success(response.content)

with st.expander("View Scraped Data"):
    st.dataframe(df)