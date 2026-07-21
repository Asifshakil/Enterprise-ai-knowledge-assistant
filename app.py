import os
import time
from datetime import datetime
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

LANGFLOW_URL = os.getenv("LANGFLOW_URL", "http://localhost:7860").rstrip("/")
FLOW_ID = os.getenv("LANGFLOW_FLOW_ID", "").strip()
API_KEY = os.getenv("LANGFLOW_API_KEY", "").strip()

APP_NAME = "Enterprise AI Knowledge Assistant"
APP_VERSION = "3.0"

DOCUMENT_COUNT = 12
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVED_CHUNKS = 8

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1380px;
            padding-top: 1.5rem;
            padding-bottom: 7rem;
        }

        section[data-testid="stSidebar"] {
            width: 245px !important;
        }

        section[data-testid="stSidebar"] > div {
            width: 245px !important;
        }

        div[data-testid="stChatMessage"] {
            border: 1px solid rgba(99, 85, 75, 0.16);
            border-radius: 16px;
            padding: 0.65rem 0.85rem;
            margin-bottom: 0.8rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(99, 85, 75, 0.16);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            min-height: 110px;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 10px;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def new_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"enterprise-ai-{timestamp}"

def extract_answer(data: dict[str, Any]) -> str:
    possible_extractors = [
        lambda value: value["outputs"][0]["outputs"][0]["results"]["message"]["text"],
        lambda value: value["outputs"][0]["outputs"][0]["artifacts"]["message"],
        lambda value: value["outputs"][0]["outputs"][0]["results"]["text"],
    ]

    for extractor in possible_extractors:
        try:
            answer = extractor(data)
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
        except (KeyError, IndexError, TypeError):
            continue

    return "The AI returned a response, but the application could not extract the answer text."

def ask_langflow(question: str) -> tuple[str, float]:
    if not FLOW_ID:
        raise RuntimeError("LANGFLOW_FLOW_ID is missing from the .env file.")
    if not API_KEY:
        raise RuntimeError("LANGFLOW_API_KEY is missing from the .env file.")

    endpoint = f"{LANGFLOW_URL}/api/v1/run/{FLOW_ID}"
    headers = {"Content-Type": "application/json", "x-api-key": API_KEY}
    payload = {
        "input_value": question,
        "input_type": "chat",
        "output_type": "chat",
        "session_id": st.session_state.session_id,
    }

    started = time.perf_counter()
    response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
    duration = time.perf_counter() - started
    response.raise_for_status()
    return extract_answer(response.json()), duration

def create_export_text() -> str:
    lines = [
        APP_NAME,
        f"Version: {APP_VERSION}",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    for message in st.session_state.messages:
        speaker = "User" if message["role"] == "user" else "Assistant"
        lines.extend([f"{speaker}:", message["content"]])

        duration = message.get("duration")
        if duration is not None:
            lines.append(f"Generated locally in {duration:.1f} seconds")

        lines.append("")

    return "\n".join(lines)

def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.session_id = new_session_id()
    st.session_state.pending_question = None

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = new_session_id()
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

with st.sidebar:
    st.title("🤖 Knowledge AI")
    st.caption("Private enterprise document intelligence")
    st.divider()

    st.subheader("System status")
    st.markdown("🟢 **Langflow API**")
    st.caption("Flow orchestration connected")
    st.markdown("🟢 **Ollama**")
    st.caption("Local language-model runtime")
    st.markdown("🟢 **ChromaDB**")
    st.caption("Persistent vector knowledge base")
    st.divider()

    st.subheader("Suggested questions")

    suggestions = [
        ("Microsoft targets", "What targets does Microsoft state for carbon, water, waste and ecosystems?"),
        ("Google priorities", "Summarize the main sustainability priorities in the Google Environmental Report."),
        ("Equinix commitments", "What sustainability commitments are described in the Equinix Sustainability Report?"),
        ("Compare strategies", "Compare Microsoft's and Google's sustainability strategies. Clearly separate each company's approach."),
    ]

    for index, (button_label, full_question) in enumerate(suggestions):
        if st.button(button_label, key=f"suggestion_{index}", use_container_width=True):
            st.session_state.pending_question = full_question
            st.rerun()

    st.divider()

    if st.session_state.messages:
        st.download_button(
            "Export conversation",
            data=create_export_text(),
            file_name="enterprise_ai_conversation.txt",
            mime="text/plain",
            use_container_width=True,
        )

    if st.button("Clear conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

    st.divider()
    st.caption("Local-first architecture\n\nDocuments and model inference remain on this computer.")

header_left, header_right = st.columns([5, 1.2], vertical_alignment="center")

with header_left:
    st.title("🤖 Enterprise AI Knowledge Assistant")
    st.write("Search, summarize and compare indexed data-center, infrastructure and sustainability reports.")

with header_right:
    st.success("● Local AI online")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

with metric_1:
    st.metric("Indexed documents", DOCUMENT_COUNT)
with metric_2:
    st.metric("Embedding model", "nomic-embed")
with metric_3:
    st.metric("Vector database", "ChromaDB")
with metric_4:
    st.metric("Local model", "Qwen 2.5 3B")

st.divider()

assistant_tab, workflow_tab, evidence_tab = st.tabs(
    ["💬 Assistant", "⚙️ AI workflow", "📋 Project evidence"]
)

with assistant_tab:
    if not st.session_state.messages:
        st.subheader("How can I help?")
        st.info("Choose a suggested question from the sidebar or type your own question below.")

        feature_1, feature_2, feature_3 = st.columns(3)

        with feature_1:
            with st.container(border=True):
                st.markdown("### 🔎 Semantic search")
                st.write("Retrieve relevant passages from indexed enterprise reports using vector similarity.")

        with feature_2:
            with st.container(border=True):
                st.markdown("### 📚 Document analysis")
                st.write("Summarize information and compare sustainability strategies across reports.")

        with feature_3:
            with st.container(border=True):
                st.markdown("### 🔒 Local processing")
                st.write("Use Ollama and ChromaDB locally without sending documents to an external model.")

        st.divider()

    st.subheader("Conversation")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("duration") is not None:
                st.caption(f"Generated locally · {message['duration']:.1f} seconds")

with workflow_tab:
    st.header("Retrieval-Augmented Generation workflow")
    st.write("This section documents the complete local AI workflow.")

    ingestion_column, question_column = st.columns(2)

    with ingestion_column:
        st.subheader("Document-ingestion workflow")
        ingestion_steps = [
            ("1", "PDF reports", "Enterprise reports are loaded into Langflow."),
            ("2", "Text splitting", f"Documents are divided into {CHUNK_SIZE}-character chunks with {CHUNK_OVERLAP}-character overlap."),
            ("3", "Embedding generation", "nomic-embed-text converts each document chunk into a vector."),
            ("4", "Vector storage", "Vectors and chunks are stored persistently in ChromaDB."),
        ]
        for number, title, description in ingestion_steps:
            with st.container(border=True):
                st.markdown(f"### {number}. {title}")
                st.write(description)

    with question_column:
        st.subheader("Question-answering workflow")
        question_steps = [
            ("1", "User question", "The user submits a question through Streamlit."),
            ("2", "Langflow API", "Streamlit sends the question through the REST API."),
            ("3", "Query embedding", "nomic-embed-text converts the question into a vector."),
            ("4", "Semantic retrieval", f"ChromaDB retrieves up to {RETRIEVED_CHUNKS} relevant chunks using MMR."),
            ("5", "Prompt assembly", "Retrieved excerpts are inserted into a grounded prompt."),
            ("6", "Local inference", "Qwen 2.5 3B generates an answer through Ollama."),
            ("7", "Final response", "Langflow returns the answer to Streamlit."),
        ]
        for number, title, description in question_steps:
            with st.container(border=True):
                st.markdown(f"### {number}. {title}")
                st.write(description)

    st.divider()
    st.subheader("Architecture diagram")
    st.code(
        """
DOCUMENT INGESTION

PDF Reports
    ↓
Read File
    ↓
Split Text
    ↓
nomic-embed-text
    ↓
ChromaDB


QUESTION ANSWERING

User Question
    ↓
Streamlit Interface
    ↓
Langflow REST API
    ↓
Question Embedding
    ↓
ChromaDB Retrieval
    ↓
Retrieved Context
    ↓
Prompt Template
    ↓
Qwen 2.5 through Ollama
    ↓
Generated Answer
        """.strip(),
        language="text",
    )

with evidence_tab:
    st.header("Project evidence")
    st.write("This section summarizes the engineering work demonstrated by the application.")

    evidence_left, evidence_right = st.columns(2)

    with evidence_left:
        with st.container(border=True):
            st.subheader("Implemented capabilities")
            st.markdown(
                """
- Local Retrieval-Augmented Generation pipeline
- Multi-document PDF ingestion
- Configurable text chunking and overlap
- Local embedding generation
- Persistent ChromaDB vector storage
- Semantic and MMR retrieval
- Prompt orchestration through Langflow
- Local Qwen inference through Ollama
- Streamlit browser interface
- Langflow REST API integration
- Conversation history and export
- Error and timeout handling
                """
            )

    with evidence_right:
        with st.container(border=True):
            st.subheader("Technology stack")
            st.markdown(
                """
- **Frontend:** Streamlit
- **Workflow orchestration:** Langflow
- **Language model:** Qwen 2.5 3B
- **Model runtime:** Ollama
- **Embedding model:** nomic-embed-text
- **Vector database:** ChromaDB
- **Programming language:** Python
- **API communication:** Requests
- **Configuration:** python-dotenv
                """
            )

    st.subheader("Portfolio summary")
    st.info(
        "Designed and implemented a local enterprise Retrieval-Augmented Generation assistant "
        "using Langflow, Ollama, ChromaDB, Qwen and Streamlit."
    )

    st.subheader("Known limitations")
    st.markdown(
        """
- Source filenames and page citations are not yet displayed.
- Retrieval quality depends on document structure and query wording.
- Qwen 2.5 3B can occasionally misclassify retrieved information.
- Response time increases when more document chunks are supplied.
        """
    )

typed_question = st.chat_input("Ask a question about the indexed reports...")
question = typed_question

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with assistant_tab:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            status = st.status("🔎 Searching the knowledge base...", expanded=True)

            try:
                status.write("🔎 Retrieving relevant passages from ChromaDB")
                status.write("📚 Preparing the retrieved document context")
                status.write("🧠 Running Qwen locally through Ollama")

                answer, duration = ask_langflow(question)

                status.write("✍️ Formatting the final response")
                status.update(label="Answer generated successfully", state="complete", expanded=False)

            except requests.exceptions.ConnectionError:
                answer = "I could not connect to Langflow. Confirm that Langflow is running."
                duration = None
                status.update(label="Could not connect to Langflow", state="error", expanded=True)

            except requests.exceptions.Timeout:
                answer = "The request timed out while waiting for the local AI pipeline."
                duration = None
                status.update(label="The request timed out", state="error", expanded=True)

            except requests.exceptions.HTTPError as error:
                answer = f"Langflow returned HTTP error {error.response.status_code}."
                duration = None
                status.update(label="Langflow returned an error", state="error", expanded=True)

            except Exception as error:
                answer = f"Application error: `{error}`"
                duration = None
                status.update(label="Application error", state="error", expanded=True)

            with st.container(border=True):
                st.markdown(answer)
                if duration is not None:
                    st.caption(f"Generated locally · {duration:.1f} seconds · Langflow + Ollama + ChromaDB")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "duration": duration}
    )

st.divider()
footer_left, footer_right = st.columns([3, 1])

with footer_left:
    st.caption(f"{APP_NAME} v{APP_VERSION} · Langflow · Ollama · ChromaDB · Streamlit")

with footer_right:
    st.caption("Local-first RAG application")
