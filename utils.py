"""
utils.py — Core AI utilities: Embeddings, FAISS search, OCR, AI response generation
Uses: sentence-transformers, FAISS, pytesseract, Anthropic API
"""

import os
import io
import json
import numpy as np
import pandas as pd
import faiss
import anthropic
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
from typing import Optional

# ─── Model Initialization ─────────────────────────────────────────────────────

# Load embedding model once (cached globally)
_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# ─── CSV Processing ────────────────────────────────────────────────────────────

def load_tickets_from_csv(file) -> pd.DataFrame:
    """
    Load Jira tickets from a CSV file.
    Tries to find common Jira column names and normalizes them.
    """
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Could not read CSV: {e}")

    # Normalize column names (lowercase, strip spaces)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Map common Jira column variations to standard names
    column_map = {
        "summary": ["summary", "title", "issue_summary", "subject"],
        "description": ["description", "desc", "details", "body", "issue_description"],
        "resolution": ["resolution", "resolved", "fix", "solution", "resolution_notes"],
        "status": ["status", "state", "ticket_status"],
        "priority": ["priority", "severity", "issue_priority"],
        "assignee": ["assignee", "assigned_to", "owner"],
        "ticket_id": ["key", "id", "ticket_id", "issue_key", "issue_id", "jira_id"],
        "created": ["created", "created_date", "date_created", "open_date"],
    }

    renamed = {}
    for standard, variants in column_map.items():
        for v in variants:
            if v in df.columns and standard not in df.columns:
                renamed[v] = standard
                break

    df = df.rename(columns=renamed)

    # Ensure minimum required columns exist
    for col in ["summary", "description", "resolution"]:
        if col not in df.columns:
            df[col] = ""

    # Fill NaN with empty string
    df = df.fillna("")

    return df


def build_ticket_text(row: pd.Series) -> str:
    """Combine key fields into a single searchable text blob."""
    parts = []
    for field in ["summary", "description", "resolution"]:
        val = str(row.get(field, "")).strip()
        if val:
            parts.append(val)
    return " | ".join(parts)


# ─── FAISS Index ───────────────────────────────────────────────────────────────

def build_faiss_index(df: pd.DataFrame):
    """
    Build a FAISS index from the ticket DataFrame.
    Returns: (faiss_index, list_of_texts, df)
    """
    model = get_embedding_model()

    # Build combined text per ticket
    texts = [build_ticket_text(row) for _, row in df.iterrows()]

    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build flat index (L2 after normalization = cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine after normalization
    index.add(embeddings)

    return index, texts, df


def search_similar_tickets(query: str, faiss_index, texts: list, df: pd.DataFrame, top_k: int = 5):
    """
    Search for top-k similar tickets to the query.
    Returns list of dicts with ticket info + similarity score.
    """
    model = get_embedding_model()

    # Encode query
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    # Search
    scores, indices = faiss_index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        row = df.iloc[idx]
        results.append({
            "similarity": round(float(score) * 100, 1),
            "ticket_id": str(row.get("ticket_id", f"TICKET-{idx}")),
            "summary": str(row.get("summary", "N/A")),
            "description": str(row.get("description", "N/A"))[:300],
            "resolution": str(row.get("resolution", "N/A"))[:400],
            "status": str(row.get("status", "Unknown")),
            "priority": str(row.get("priority", "Unknown")),
        })

    return results


# ─── OCR ───────────────────────────────────────────────────────────────────────

def extract_text_from_image(image_file) -> str:
    """
    Extract text from a screenshot/image using pytesseract OCR.
    Returns extracted text string.
    """
    try:
        image = Image.open(image_file)

        # Improve OCR quality: convert to grayscale
        image = image.convert("L")

        # Run OCR
        text = pytesseract.image_to_string(image, config="--psm 6")
        text = text.strip()

        if not text:
            return "⚠️ No text could be extracted from the image."
        return text

    except Exception as e:
        return f"❌ OCR failed: {str(e)}"


# ─── AI Response Generation ────────────────────────────────────────────────────

def generate_ai_response(
    ticket_text: str,
    similar_tickets: list,
    api_key: str
) -> dict:
    """
    Call Anthropic Claude API to generate structured ticket resolution guidance.
    Returns a dict with: root_cause, steps, troubleshooting, preventive, jira_comment, confidence
    """
    client = anthropic.Anthropic(api_key=api_key)

    # Build context from similar tickets
    similar_context = ""
    for i, t in enumerate(similar_tickets[:3], 1):
        similar_context += f"\n[Similar Ticket {i} — {t['similarity']}% match]\n"
        similar_context += f"Summary: {t['summary']}\n"
        similar_context += f"Resolution: {t['resolution']}\n"

    system_prompt = """You are an expert L2/L3 IT support engineer with deep knowledge of 
DevOps, cloud infrastructure, Jira workflows, and enterprise software platforms.
You analyze support tickets and provide structured, actionable resolution guidance.
Always respond in valid JSON format exactly as specified."""

    user_prompt = f"""Analyze this IT support ticket and provide a structured resolution guide.

TICKET:
{ticket_text}

SIMILAR RESOLVED TICKETS FOR CONTEXT:
{similar_context if similar_context else "No similar tickets found in knowledge base."}

Respond ONLY with a valid JSON object (no markdown, no extra text) in this exact format:
{{
  "root_cause": "Clear explanation of what likely caused this issue (2-3 sentences)",
  "solution_steps": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ],
  "troubleshooting_steps": [
    "Check 1: ...",
    "Check 2: ...",
    "Check 3: ..."
  ],
  "preventive_measures": [
    "Prevention 1: ...",
    "Prevention 2: ..."
  ],
  "jira_comment": "Ready-to-paste Jira comment with greeting, diagnosis summary, resolution steps, and next action",
  "confidence": 85,
  "category": "Infrastructure / Network / Access / Application / Database / Other",
  "estimated_resolution_time": "e.g. 30 minutes"
}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        raw = response.content[0].text.strip()

        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()

        result = json.loads(raw)
        return result

    except json.JSONDecodeError:
        # Return graceful fallback
        return {
            "root_cause": "AI could not parse a structured response. Please review the ticket manually.",
            "solution_steps": ["Review the ticket description carefully.", "Check similar tickets in the knowledge base.", "Escalate to L2 if unresolved."],
            "troubleshooting_steps": ["Verify the issue is reproducible.", "Check system logs.", "Contact the affected user for more details."],
            "preventive_measures": ["Document resolution steps.", "Update runbook if needed."],
            "jira_comment": f"Hi, I've reviewed your ticket. I'm investigating the issue and will update you shortly.\n\nTicket details: {ticket_text[:200]}...\n\nNext steps: Under investigation. Will update within 2 hours.",
            "confidence": 0,
            "category": "Unknown",
            "estimated_resolution_time": "TBD"
        }
    except Exception as e:
        return {
            "error": str(e),
            "root_cause": f"API Error: {str(e)}",
            "solution_steps": ["Check your API key.", "Retry the request."],
            "troubleshooting_steps": [],
            "preventive_measures": [],
            "jira_comment": "Under investigation.",
            "confidence": 0,
            "category": "Unknown",
            "estimated_resolution_time": "TBD"
        }


# ─── Chatbot ───────────────────────────────────────────────────────────────────

def chat_with_tickets(
    user_message: str,
    chat_history: list,
    faiss_index,
    texts: list,
    df: pd.DataFrame,
    api_key: str
) -> str:
    """
    Chatbot that answers questions about tickets using RAG.
    Maintains conversation history.
    """
    client = anthropic.Anthropic(api_key=api_key)

    # Find relevant tickets for context
    relevant = search_similar_tickets(user_message, faiss_index, texts, df, top_k=3)
    context = ""
    for t in relevant:
        context += f"\n- [{t['ticket_id']}] {t['summary']} | Resolution: {t['resolution'][:200]}\n"

    # Build message history for API
    messages = []
    for h in chat_history[-10:]:  # Keep last 10 turns to manage context
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_message})

    system_prompt = f"""You are a helpful IT support chatbot with access to a knowledge base of resolved Jira tickets.
You help support engineers quickly find solutions and understand ticket patterns.
Be concise, practical, and technical. Use bullet points for steps.

RELEVANT TICKET CONTEXT:
{context if context else "No specific tickets matched this query."}

Answer questions based on the ticket data. If asked about specific tickets, refer to the context provided."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text

    except Exception as e:
        return f"❌ Chatbot error: {str(e)}"
