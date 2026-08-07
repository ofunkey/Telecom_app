"""
Builds a merged retriever across all three Chroma collections:
  - faq     : FAQ entries (no chunking — 1 row = 1 doc)
  - tickets : resolved support tickets (no chunking — 1 ticket = 1 doc)
  - guides  : PDF guide chunks (RecursiveCharacterTextSplitter applied at ingest)
Returns:
    list[Document] where each Document contains:
        metadata["source"]
        metadata["score"]   # similarity distance from Chroma

"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

CHROMA_DIR  = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# add similarity threshold, Chroma returns a DISTANCE score, lower distance = better match in Chroma
# Chroma returns a DISTANCE score:
#
#     0.00  = identical
#     0.20  = very similar
#     0.50  = good
#     0.80+ = weak
#
# Lower is better.
#
# ---------------------------------------------------------------------
SIMILARITY_THRESHOLD = 0.75


def build_retriever(
    k_faq: int = 3,
    k_tickets: int = 3,
    k_guides: int = 3,
) -> RunnableLambda:
    
    """
    Returns a retrieval function that:

    1. Searches all three Chroma collections.
    2. Retrieves similarity scores.
    3. Filters out documents below the confidence threshold.
    4. Returns the remaining Documents sorted by best match.
    """    
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    faq_store = Chroma(
        collection_name="faq",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    guides_store = Chroma(
        collection_name="guides",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # faq_retriever     = faq_store.as_retriever(search_kwargs={"k": k_faq})
    # tickets_retriever = tickets_store.as_retriever(search_kwargs={"k": k_tickets})
    # guides_retriever  = guides_store.as_retriever(search_kwargs={"k": k_guides})

    # def retrieve(query: str) -> list[Document]:
    #     return (
    #         faq_retriever.invoke(query)
    #         + tickets_retriever.invoke(query)
    #         + guides_retriever.invoke(query)
    #     )


    def retrieve(query: str) -> list[Document]:
        """
        Search all collections and return only
        high-confidence documents.
        """
        results = []

        # ---------------------------------------------------------
        # FAQ
        # ---------------------------------------------------------
        results.extend(
            faq_store.similarity_search_with_score(query, k=k_faq)
        )

        # ---------------------------------------------------------
        # Tickets
        # ---------------------------------------------------------
        results.extend(
            tickets_store.similarity_search_with_score(query, k=k_tickets)
        )

        # ---------------------------------------------------------
        # Guides
        # ---------------------------------------------------------
        results.extend(
            guides_store.similarity_search_with_score(query, k=k_guides)
        )

        # No results at all
        if not results:
            return []

        # ----------------------------------------------------
        # Confidence Check
        # ----------------------------------------------------
        best_score = min(score for _, score in results)

        # Best match isn't good enough
        if best_score > SIMILARITY_THRESHOLD:
            return []
        # Keep only documents whose distance is below threshold
        docs = []

        for doc, score in results:
            # save score for later inspection
            doc.metadata["score"] = score
            docs.append(doc)
        docs.sort(key=lambda d: d.metadata["score"])
        return docs

    return RunnableLambda(retrieve)
