import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.vectorstores import  FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


from dotenv import load_dotenv
import os
from langchain_core.documents import Document


load_dotenv()
API_KEY = os.getenv('Langchain')

st.set_page_config(page_title="Laptop AI Assistant",layout="wide")
st.title("Laptop AI Assistant with Web Scrapping")

@st.cache_data
def scrape_laptop_data():
    url = "https://webscraper.io/test-sites/e-commerce/static/computers/laptops?page="
    url_lst = []
    for i in range(1,21):
        new_url = url+str(i)
        url_lst.append(new_url)
    res =[]
    for u in url_lst:
        response = requests.get(u).text
        soup_obj = BeautifulSoup(response,'html.parser')
        all_div = soup_obj.find_all('div',class_='col-md-4 col-xl-4 col-lg-4') #20 pages -- per page 6 
        for d in all_div:
            product_name = d.find('a',class_='title').text.strip()
            product_price = d.find('span').text
            description = d.find('p',class_='description card-text').text
            review = d.find('p',class_='review-count float-end').text.strip()
            product_data = {
            "Product Name":product_name,
            "Product Price":product_price,
            "Description":description,
            "Review":review
            }
            res.append(product_data)
    df = pd.DataFrame(res)
    return df

df = scrape_laptop_data()

def row_to_text(row):
    return f"""
    Product Name : {row['Product Name']}
    Product Price : {row['Product Price']}
    Description : {row['Description']}
    Review : {row['Review']}
    """

# documents = df.apply(row_to_text,axis=1).to_list()

# print(59,type(documents))

documents = [Document(page_content=row_to_text(r)) for _,r in df.iterrows()]

embeddings =  OpenAIEmbeddings(model='text-embedding-3-small',api_key=API_KEY)
vectorstore = FAISS.from_documents(documents,embeddings)
llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)

prompt = PromptTemplate(
    input_variable=["context","question"],
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
    if any(x in q for x in ["list","show","all","below","under","less than"]):
        return "FILTER"
    if any(x in q for x in ["best","recommended","suggest"]):
        return "RECOMMEND"
    return "FACT"

def extract_price(query):
    match = re.search(r"\$?(\d+)",query)
    return float(match.group(1)) if match else None


def extract_review(query):
    query = query.lower()
    match = re.search(r"(\d+)\s*reviews?",query)
    return int(match.group(1)) if match else 0


query = st.text_input("Ask a Question About Laptops: ")

if query:
    qtype = detect_question_type(query)
    st.caption(f"Detected Question Type is {qtype}")

    if qtype == "FACT":
        docs = vectorstore.similarity_search(query,k=4)
        context_text = "\n".join(d.page_content for d in docs)
        response = chain.invoke({
            "context":context_text,
            "question":query
        })
        st.success(response.content)
    elif qtype=="FILTER":
        price_limit = extract_price(query)
        print("118",price_limit,type(price_limit))
        print(df.info())
        if price_limit is None:
            st.warning("Please set a Price Range.")
        else:
            df['Product Price'] = (
                df['Product Price']
                .astype(str).str.replace('$','',regex=False)
                .astype(float)
            )
            filtered = df[df['Product Price']<price_limit]

            if filtered.empty:
                st.warning("No laptop under this criteria.")
            else:
                docs = filtered.apply(row_to_text,axis=1).to_list()
                context_text = "\n".join(docs[:10])
                response = chain.invoke({
                    "context":context_text,
                    "question":"List the laptop names with prices"
                })

                st.success(response.content)
    elif qtype == "RECOMMEND":
        price_limit = extract_price(query) or 1000
        print(148,price_limit)
        df['Product Price'] = (
                df['Product Price']
                .astype(str).str.replace('$','',regex=False)
                .astype(float)
            )
        
        ex_review = extract_review(query)
        print(156,ex_review)

        df['Review'] = (
            df['Review']
                .astype(str).str.replace('reviews','',regex=False)
                .astype(int)
        )

        print(df['Review'])


        filtered = df[
            (df['Product Price']>1000) & (df['Review']>=ex_review)
        ]

        print(159,filtered)

        if filtered.empty:
            st.warning("No suitable Laptop found")
        else:
            print("inside else")
            docs = filtered.apply(row_to_text,axis=1).to_list()
            print(166,docs)
            context_text = "\n".join(docs[:10])
            response = chain.invoke({
                    "context":context_text,
                    "question":"List the laptop names with prices and reviews"
                })
            st.success(response.content)

with st.expander("View Your Data"):
    st.dataframe(df)

