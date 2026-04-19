"""
Evaluation Script — Run RAGAS metrics and print a report.

Usage:
    python tests/evaluate.py

Requirements:
    pip install ragas datasets
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.rag_pipeline import RAGPipeline
from datasets import Dataset


# ── Test dataset ───────────────────────────────────────────────────────────────
# Add your own question/ground_truth pairs here for accurate evaluation.
TEST_SET = [
    {
        "question": "How does the RAG chatbot work?",
        "ground_truth": "RAG combines large language models with a searchable knowledge base using vector similarity search to retrieve relevant chunks, then generates grounded answers.",
    },
    {
        "question": "What embedding model is used?",
        "ground_truth": "OpenAI's text-embedding-3-small model is used for generating embeddings.",
    },
    {
        "question": "How do I add my own documents?",
        "ground_truth": "Use the /upload endpoint to add PDF, TXT, or Markdown files.",
    },
    {
        "question": "What evaluation metrics are used?",
        "ground_truth": "RAGAS metrics including faithfulness, answer relevancy, and context recall are used.",
    },
]


def run_evaluation():
    print("\n" + "="*60)
    print("  RAG Chatbot — RAGAS Evaluation")
    print("="*60)

    # Initialize pipeline
    print("\n[1/3] Loading RAG pipeline...")
    rag = RAGPipeline()
    rag.load_or_build_vectorstore()

    # Build evaluation rows
    print("[2/3] Running queries...")
    rows = []
    for item in TEST_SET:
        result = rag.chain.invoke({"query": item["question"]})
        rows.append({
            "question": item["question"],
            "answer": result["result"],
            "contexts": [d.page_content for d in result["source_documents"]],
            "ground_truth": item["ground_truth"],
        })
        print(f"  ✓ '{item['question'][:50]}...'")

    # Evaluate
    print("[3/3] Computing RAGAS metrics...")
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )

        dataset = Dataset.from_list(rows)
        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        )

        print("\n" + "="*60)
        print("  Results")
        print("="*60)
        for metric, score in scores.items():
            bar = "█" * int(float(score) * 20)
            print(f"  {metric:<25} {float(score):.4f}  {bar}")

        print("\n  → Add these numbers to your resume!")
        print("="*60 + "\n")

    except ImportError:
        print("\n⚠️  Install ragas to run evaluation:")
        print("   pip install ragas datasets\n")

    except Exception as e:
        print(f"\n⚠️  Evaluation error: {e}\n")


if __name__ == "__main__":
    run_evaluation()
