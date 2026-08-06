"""
Builds the RAG chain

Flow

Question
    │
    ▼
Merged Retriever
    │
    ▼
Confidence Filter
    │
    ├────────────► No documents
    │                  │
    │                  ▼
    │         Return canned response
    │
    ▼
Format Context
    │
    ▼
Prompt
    │
    ▼
Qwen3-27B on Groq
    │
    ▼
String Output

  question → merged retriever → confidence filter → prompt → Qwen3-27B on Groq → string output
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
import os
import requests

from langchain_groq import ChatGroq

from retriever import build_retriever

FALLBACK_RESPONSE = """
I'm sorry, but I couldn't find enough information in our knowledge base to answer your question confidently.

Please contact Telecom Customer Support by dialing **611**, or use the **MyTelecom** app for additional assistance.
"""

# SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
# Your job is to help customers resolve technical issues with their mobile service.

# Use ONLY the context below to answer the customer's question.
# The context comes from two sources:
# - FAQ entries (general policy and how-to information)
# - Past support tickets (real resolved cases with step-by-step resolutions)

# If the context does not contain enough information to answer confidently, say so clearly \
# and suggest the customer call 611 or use the MyTelecom app.

# Context:
# {context}
# """

# updated SYSTEM_PROMPT to include source of context
SYSTEM_PROMPT = """
You are a helpful and professional telecom customer care assistant.

Use ONLY the supplied context.

The context comes from:

- FAQ entries
- Support tickets
- User guides

Rules:

Every factual statement must include the citation from the retrieved context.

Examples

(Source: FAQ)

(Source: Ticket ID: TICKET-1044)

(Source: Guide ID: TELECOM_GUIDE, Page 3)

Never invent ticket IDs, guide IDs, or page numbers.
Only use the metadata provided in the context.

If the answer is not supported by context, say so.

Context

{context}
 
 Conversational rules:
 - Be conversational and allow the user to ask follow-up questions.
 - If the context fully answers the user's question, answer directly and do not ask a clarifying question.
 - If the context does not fully answer the user's question, ask ONE concise clarifying question instead of guessing.
 - When the user describes a symptom or issue, prefer a specific follow-up question rather than a generic prompt.
 - For dropped calls, roaming, or connectivity issues, ask a narrow clarifying question such as:
   "Are the drops happening only in one location, while roaming, or when you’re on Wi-Fi?"
 - For network speed or data issues, ask whether the problem occurs on all apps, one app, or only in certain areas.
 - If you can answer from context, answer clearly and then optionally add a brief closing offer: "If you want, I can explain that in more detail."
 - Do not invent information or ticket/guide IDs; cite sources from the context.
"""


# def _format_docs(docs: list[Document]) -> str:
#     sections = []
#     for doc in docs:
#         source = doc.metadata.get("source", "unknown").upper()
#         sections.append(f"[{source}]\n{doc.page_content}")
#     return "\n\n---\n\n".join(sections)


# updated _format_docs to include source label and citation format
def _format_docs(docs: list[Document]) -> str:

    sections = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown").upper()

        if source == "GUIDE":
            guide_id = doc.metadata.get("guide_id", "UNKNOWN_GUIDE")
            page = doc.metadata.get("page_label", "?")

            header = f"[GUIDE | Guide ID: {guide_id} | Page: {page}]"

        elif source == "TICKET":
            ticket_id = doc.metadata.get("ticket_id", "UNKNOWN_TICKET")

            header = f"[TICKET | Ticket ID: {ticket_id}]"

        elif source == "FAQ":
            faq_id = doc.metadata.get("faq_id")

            if faq_id:
                header = f"[FAQ | FAQ ID: {faq_id}]"
            else:
                header = "[FAQ]"

        else:
            header = "[UNKNOWN]"

        sections.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(sections)


def build_chain():
    retriever = build_retriever()

    # Include chat history in the human message so the model can be conversational.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{chat_history}\nUser: {question}"),
        ]
    )

    # Allow the model name to be configured via environment variable so
    # users can correct the model identifier or switch to an accessible
    # model without editing code.
    model_name = os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b")
    print(f"[rag_chain] Using GROQ_MODEL={model_name}")

    llm = ChatGroq(
        model=model_name,
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

    # Wrap the Groq LLM invocation so failures (e.g. model not found or
    # permission errors) don't crash the application.
    # If OPENAI_API_KEY is set, we will use OpenAI only as a secondary fallback.
    # If OPENAI_API_KEY is unset, we keep the no-charge Groq-only behavior.
    def _safe_llm_invoke(prompt_input):
        try:
            return llm.invoke(prompt_input)
        except Exception as e:
            import traceback, sys

            traceback.print_exc()
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                try:
                    if isinstance(prompt_input, dict):
                        user_text = prompt_input.get("question", "")
                        history = prompt_input.get("chat_history", "")
                    else:
                        user_text = str(prompt_input)
                        history = ""

                    openai_prompt = f"{history}\nUser: {user_text}"
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {openai_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": openai_prompt},
                            ],
                            "temperature": 0,
                            "max_tokens": 512,
                        },
                        timeout=20,
                    )
                    data = response.json()
                    if (
                        response.status_code == 200
                        and "choices" in data
                        and data["choices"]
                    ):
                        return data["choices"][0]["message"]["content"]
                except Exception:
                    traceback.print_exc()
                    # fall through to the default fallback message below

            guidance = (
                "The configured Groq model is not available or you do not have access.\n"
                "Check the `GROQ_MODEL` environment variable and your Groq API key/permissions.\n"
                "Example (PowerShell): $env:GROQ_MODEL = 'your-valid-model'\n"
                "Set your Groq API key as required by your environment (check Groq docs)."
            )
            return FALLBACK_RESPONSE + "\n\n" + guidance + f"\n\nDebug: {e}"

    safe_llm = RunnableLambda(_safe_llm_invoke)

    parser = StrOutputParser()

    # chain = (
    #     {"context": retriever | _format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | parser
    # )
    # return chain

    # ---------------------------------------------------
    # Step 1
    # Retrieve documents while keeping the question
    # ---------------------------------------------------

    # accept either a raw question string or a dict with question + chat_history
    retrieve = RunnableLambda(
        lambda x: {
            "question": x["question"] if isinstance(x, dict) else x,
            "docs": retriever.invoke(x["question"] if isinstance(x, dict) else x),
            "chat_history": x.get("chat_history", "") if isinstance(x, dict) else "",
        }
    )

    # ---------------------------------------------------
    # Step 2
    # Build prompt inputs
    # ---------------------------------------------------

    build_prompt_inputs = RunnableLambda(
        lambda x: {
            "question": x["question"],
            "context": _format_docs(x["docs"]),
            "chat_history": x.get("chat_history", ""),
        }
    )

    # ---------------------------------------------------
    # Step 3
    # LLM chain
    # ---------------------------------------------------

    llm_chain = build_prompt_inputs | prompt | safe_llm | parser

    # ---------------------------------------------------
    # Step 4
    # Fallback chain
    # ---------------------------------------------------

    fallback_chain = RunnableLambda(lambda _: FALLBACK_RESPONSE)

    # ---------------------------------------------------
    # Step 5
    # Confidence router
    # ---------------------------------------------------

    router = RunnableBranch(
        # If no documents survived the confidence filter
        (
            lambda x: len(x["docs"]) == 0,
            fallback_chain,
        ),
        # Otherwise continue to the LLM
        llm_chain,
    )

    # ---------------------------------------------------
    # Final chain
    # ---------------------------------------------------

    chain = RunnablePassthrough() | retrieve | router

    return chain
