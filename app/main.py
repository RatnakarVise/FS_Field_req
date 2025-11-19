from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import os, json, gc
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Load environment
# ------------------------------------------------------------------
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

os.environ["LANGCHAIN_TRACING_V2"] = "true"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

app = FastAPI(title="ABAP Field Logic Extractor API")

# ------------------------------------------------------------------
# Input Model
# ------------------------------------------------------------------
class ABAPSnippet(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    code: str

    @field_validator("code", mode="before")
    @classmethod
    def clean_code(cls, v):
        return v.strip() if v else v


# ------------------------------------------------------------------
# Summarizer
# ------------------------------------------------------------------
def snippet_json(snippet: ABAPSnippet) -> dict:
    return {
        "pgm_name": snippet.pgm_name,
        "inc_name": snippet.inc_name,
        "unit_type": snippet.type,
        "unit_name": snippet.name,
        "code": snippet.code,
    }


# ------------------------------------------------------------------
# Load RAG Knowledge Base
# ------------------------------------------------------------------
rag_file_path = os.path.join(os.path.dirname(__file__), "rag_knowledge_base.txt")
loader = TextLoader(file_path=rag_file_path, encoding="utf-8")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=200)
docs = splitter.split_documents(documents)

# Build vector store
embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()


# ------------------------------------------------------------------
# Chain builder
# ------------------------------------------------------------------
def build_chain(snippet: ABAPSnippet):
    retrieved_docs = retriever.invoke(snippet.code)
    retrieved_context = "\n\n".join([d.page_content for d in retrieved_docs])

    SYSTEM_MSG = """
You are a senior SAP ABAP analyst.
Your goal is to extract ONLY meaningful business fields involved in data flow.

DO NOT extract:
- work areas (WA_…)
- internal tables (IT_…)
- local variables(LV_..)
- loop counters
- substring expressions (MATNR+14(1))
- SY-* fields (unless used in final business logic)
- message variables
- flags (CHECK, CHECK1, FLG, etc.)
- helper variables (HEADER, MESSAGEPART, etc.)
- if no field create blank

YOU MUST extract ONLY those fields that:
1. Come from database SELECT/SELECT SINGLE/FOR ALL ENTRIES
2. Are used to derive business values
3. Affect final output or BOM creation logic
4. Affect filtering, grouping, or validation

For each field:
- Consolidate logic from ALL occurrences across the snippet
- explaination must be in functional/business language with all the requiired conditions
- Combine logic from SELECT, WHERE, LOOP, IF, calculations
- Return MERGED logic, not individual fragments

Output STRICT JSON ONLY:
[
  {{
    "field_name": "<field>",
    "field_logic": "<complete derived logic>"
  }}
]
    """

    USER_TEMPLATE = """
Analyze the ABAP snippet and list **every field** with logic details.

Program: {pgm_name}
Include: {inc_name}
Unit type: {type}
Unit name: {name}

RAG Knowledge Base Context:
{retrieved_context}

Snippet JSON:
{context_json}

IMPORTANT:
- Do NOT add extra text.
- Return ONLY JSON.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ])

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    parser = JsonOutputParser()

    return prompt | llm | parser, retrieved_context


# ------------------------------------------------------------------
# LLM Runner
# ------------------------------------------------------------------
def run_llm(snippet: ABAPSnippet):
    ctx_json = json.dumps(snippet_json(snippet), ensure_ascii=False, indent=2)
    chain, retrieved_context = build_chain(snippet)

    return chain.invoke({
        "context_json": ctx_json,
        "pgm_name": snippet.pgm_name,
        "inc_name": snippet.inc_name,
        "type": snippet.type,
        "name": snippet.name,
        "retrieved_context": retrieved_context,
    })


# ------------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------------
@app.post("/extract-field-logic")
async def extract_field_logic(snippets: List[ABAPSnippet]):
    results = []

    for snippet in snippets:
        try:
            llm_result = run_llm(snippet)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM error: {e}")

        results.append({
            "pgm_name": snippet.pgm_name,
            "inc_name": snippet.inc_name,
            "type": snippet.type,
            "name": snippet.name,
            "field_logic": llm_result,
        })

    return results


@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
