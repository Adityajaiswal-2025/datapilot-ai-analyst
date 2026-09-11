import pandas as pd
from typing import Optional, List, Dict, Any
from app.tools.profiling import profile_dataset, ProfilingError
from app.data.metadata import get_registered_dataset
from app.schemas.dataset import ColumnProfile, DatasetProfile



class ContextGenerationError(Exception):
    """Custom exception raised when context formatting fails."""
    pass


def build_column_context(col: ColumnProfile) -> str:
    """Formats a single column profile into a compact, high-density context line."""
    base = f"- Column '{col.name}' ({col.classified_type}, {col.raw_dtype}) | Nulls: {col.null_count} ({col.null_percentage}%) | Unique: {col.unique_count} ({col.unique_percentage}%)"

    if col.is_constant:
        base += " [CONSTANT COLUMN]"

    if col.classified_type == "numeric" and col.numeric_stats:
        ns = col.numeric_stats
        base += f" | Stats: min={ns.min}, max={ns.max}, mean={ns.mean}, median={ns.median}, std={ns.std}"

    elif col.classified_type in ["categorical", "text", "boolean"] and col.categorical_stats:
        cs = col.categorical_stats
        mode_str = f"'{cs.mode}'" if cs.mode is not None else "N/A"
        top_items = ", ".join([f"{item['value']} ({item['count']})" for item in cs.top_frequent_values[:3]])
        base += f" | Mode: {mode_str} | Top Values: [{top_items}]"

    elif col.classified_type == "datetime" and col.datetime_stats:
        ds = col.datetime_stats
        base += f" | Range: {ds.min_date} to {ds.max_date} ({ds.date_range_days} days)"

    return base


def format_sample_preview(sample_rows: List[Dict[str, Any]], max_rows: int = 3) -> str:
    """Formats sample data rows into clean text lines for LLM prompts."""
    if not sample_rows:
        return "No sample rows available."

    lines: List[str] = []
    for idx, row in enumerate(sample_rows[:max_rows], start=1):
        row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
        lines.append(f"Row {idx}: {{ {row_str} }}")

    return "\n".join(lines)


def build_dataset_context(
    dataset_id: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    profile: Optional[DatasetProfile] = None,
    max_sample_rows: int = 3,
    query: Optional[str] = None,
    include_rag_context: bool = True,
) -> str:
    """Generates a structured, LLM-optimized context prompt string for a dataset."""
    if profile is None:
        if not dataset_id and df is None:
            raise ContextGenerationError("Either dataset_id, df, or profile must be provided to build_dataset_context().")
        try:
            profile = profile_dataset(dataset_id=dataset_id, df=df)
        except ProfilingError as e:
            raise ContextGenerationError(str(e)) from e

    qr = profile.quality_report

    context_parts: List[str] = []

    # Section 1: Overview
    context_parts.append(
        f"### DATASET OVERVIEW\n"
        f"- Dataset ID: {profile.dataset_id}\n"
        f"- Filename: {profile.filename}\n"
        f"- Dimensions: {profile.row_count} rows × {profile.column_count} columns\n"
        f"- Quality Score: {qr.quality_score}/100 (Completeness: {qr.completeness_percentage}%, Duplicates: {qr.duplicate_rows_count} rows [{qr.duplicate_rows_percentage}%])"
    )

    # Section 2: Quality Warnings
    if qr.warnings:
        warn_lines = "\n".join([f"- WARNING: {w}" for w in qr.warnings])
        context_parts.append(f"### DATA QUALITY WARNINGS\n{warn_lines}")
    else:
        context_parts.append("### DATA QUALITY WARNINGS\n- No critical data quality issues detected.")

    # Section 3: Column Breakdown
    col_lines = [build_column_context(col) for col in profile.column_profiles]
    context_parts.append("### COLUMN SCHEMA & STATISTICS\n" + "\n".join(col_lines))

    # Section 4: Sample Rows Preview
    entry = get_registered_dataset(profile.dataset_id) if profile.dataset_id else None
    sample_rows = []
    if entry and "metadata" in entry:
        sample_rows = entry["metadata"].sample_rows

    if sample_rows:
        sample_text = format_sample_preview(sample_rows, max_rows=max_sample_rows)
        context_parts.append(f"### DATA SAMPLE (Top {min(len(sample_rows), max_sample_rows)} Rows)\n{sample_text}")

    # Section 5: RAG Domain Knowledge & Glossary Integration
    if include_rag_context:
        try:
            from app.rag import default_rag_retriever
            search_query = query
            if not search_query and profile.column_profiles:
                search_query = " ".join([c.name for c in profile.column_profiles[:5]])

            if search_query:
                rag_results = default_rag_retriever.search(query=search_query, top_k=3)
                if rag_results:
                    rag_lines = []
                    for item in rag_results:
                        rag_lines.append(f"- [{item['doc_title']}]: {item['content']}")
                    context_parts.append("### DOMAIN KNOWLEDGE & GLOSSARY (RAG)\n" + "\n".join(rag_lines))
        except Exception:
            pass

    return "\n\n".join(context_parts)
