import os
import uuid
import base64
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server generation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, Dict, Any
from app.tools.analysis import _resolve_dataframe, AnalysisToolError


CHARTS_DIR = "data/static/charts"


def generate_visualization(
    dataset_id: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    chart_type: str = "bar",
    x_column: str = "",
    y_column: Optional[str] = None,
    title: Optional[str] = None,
    save_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates a visualization chart using Matplotlib/Seaborn and saves PNG file to disk.

    Returns chart metadata, file path, and base64 encoded image string.
    """
    target_df = _resolve_dataframe(dataset_id, df)

    if not x_column or x_column not in target_df.columns:
        raise AnalysisToolError(f"X-axis column '{x_column}' not found in dataset.")

    if y_column and y_column not in target_df.columns:
        raise AnalysisToolError(f"Y-axis column '{y_column}' not found in dataset.")

    os.makedirs(CHARTS_DIR, exist_ok=True)
    chart_id = save_filename or f"chart_{uuid.uuid4().hex[:8]}.png"
    if not chart_id.endswith(".png"):
        chart_id += ".png"
    file_path = os.path.join(CHARTS_DIR, chart_id)

    # Set aesthetic style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    chart_type_lower = chart_type.lower()

    try:
        if chart_type_lower == "bar":
            if y_column:
                # Aggregate top 15 categories for clean presentation
                sub = target_df.groupby(x_column, as_index=False)[y_column].mean().head(15)
                sns.barplot(data=sub, x=x_column, y=y_column, hue=x_column, palette="viridis", legend=False)
            else:
                counts = target_df[x_column].value_counts().head(15).reset_index()
                counts.columns = [x_column, "count"]
                sns.barplot(data=counts, x=x_column, y="count", hue=x_column, palette="viridis", legend=False)
            plt.xticks(rotation=45, ha="right")

        elif chart_type_lower == "line":
            if not y_column:
                raise AnalysisToolError("Line chart requires both x_column and y_column.")
            sns.lineplot(
                data=target_df,
                x=x_column,
                y=y_column,
                marker="o",
                color="#2b5c8f",
                linewidth=2.5,
            )
            plt.xticks(rotation=45, ha="right")

        elif chart_type_lower == "scatter":
            if not y_column:
                raise AnalysisToolError("Scatter chart requires both x_column and y_column.")
            sns.scatterplot(
                data=target_df,
                x=x_column,
                y=y_column,
                color="#e74c3c",
                alpha=0.8,
                s=70,
            )

        elif chart_type_lower == "histogram":
            sns.histplot(data=target_df, x=x_column, kde=True, color="#2ecc71", bins=20)

        elif chart_type_lower == "boxplot":
            if y_column:
                sns.boxplot(data=target_df, x=x_column, y=y_column, hue=x_column, palette="Set2", legend=False)
                plt.xticks(rotation=45, ha="right")
            else:
                sns.boxplot(data=target_df, y=x_column, color="#3498db")


        elif chart_type_lower == "pie":
            counts = target_df[x_column].value_counts().head(7)
            plt.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
                startangle=140,
                colors=sns.color_palette("pastel"),
            )

        else:
            raise AnalysisToolError(
                f"Unsupported chart type '{chart_type}'. Supported: 'bar', 'line', 'scatter', 'histogram', 'boxplot', 'pie'."
            )

        chart_title = (
            title
            or f"{chart_type.capitalize()} Chart: {x_column}"
            + (f" vs {y_column}" if y_column else "")
        )
        plt.title(chart_title, fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(file_path, format="png", dpi=150)
        plt.close()

        # Read base64 string
        with open(file_path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("utf-8")

        return {
            "chart_type": chart_type_lower,
            "x_column": x_column,
            "y_column": y_column,
            "title": chart_title,
            "file_path": file_path,
            "base64": f"data:image/png;base64,{b64_str}",
        }

    except Exception as e:
        plt.close()
        raise AnalysisToolError(f"Failed to render chart: {str(e)}") from e
