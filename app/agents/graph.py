import logging
import pandas as pd
from typing import TypedDict, Optional, List, Dict, Any, Literal
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph.graph import StateGraph, START, END

from app.schemas.agents import (
    SupervisorOutput,
    ProfilerOutput,
    AnalystOutput,
    VisualizationOutput,
    InsightOutput,
)
from app.agents.supervisor import run_supervisor_agent
from app.agents.profiler import run_profiler_agent
from app.agents.analyst import run_analyst_agent
from app.agents.visualization import run_visualization_agent
from app.agents.insight import run_insight_agent
from app.memory.manager import rewrite_contextual_query

logger = logging.getLogger("datapilot.agents.graph")


class DataPilotState(TypedDict, total=False):
    """Shared conversational state passed across LangGraph agent nodes."""
    query: str
    session_id: Optional[str]
    dataset_id: Optional[str]
    df: Optional[pd.DataFrame]
    llm: Optional[BaseChatModel]
    next_agent: str
    supervisor_output: Optional[SupervisorOutput]
    profiler_output: Optional[ProfilerOutput]
    analyst_output: Optional[AnalystOutput]
    vis_output: Optional[VisualizationOutput]
    insight_output: Optional[InsightOutput]
    errors: List[str]
    history: List[str]


def supervisor_node(state: DataPilotState) -> Dict[str, Any]:
    """Supervisor routing node."""
    raw_query = state.get("query", "")
    session_id = state.get("session_id")
    dataset_id = state.get("dataset_id")
    df = state.get("df")
    llm = state.get("llm")

    # Contextual Query Rewriting via Session Memory
    query = rewrite_contextual_query(query=raw_query, session_id=session_id, llm=llm)

    sup_out = run_supervisor_agent(query=query, dataset_id=dataset_id, df=df, llm=llm)
    history = list(state.get("history", []))
    history.append(f"Supervisor routed request to node '{sup_out.next_agent}'")

    return {
        "query": query,
        "supervisor_output": sup_out,
        "next_agent": sup_out.next_agent,
        "history": history,
    }


def profiler_node(state: DataPilotState) -> Dict[str, Any]:
    """Data Profiler Agent node."""
    dataset_id = state.get("dataset_id")
    df = state.get("df")
    llm = state.get("llm")

    prof_out = run_profiler_agent(dataset_id=dataset_id, df=df, llm=llm)
    history = list(state.get("history", []))
    history.append("Profiler agent node completed dataset profiling")

    return {
        "profiler_output": prof_out,
        "next_agent": "end",
        "history": history,
    }


def analyst_node(state: DataPilotState) -> Dict[str, Any]:
    """Data Analyst Agent node."""
    query = state.get("query", "")
    dataset_id = state.get("dataset_id")
    df = state.get("df")
    llm = state.get("llm")

    analyst_out = run_analyst_agent(query=query, dataset_id=dataset_id, df=df, llm=llm)
    history = list(state.get("history", []))
    history.append("Analyst agent node completed quantitative analysis")

    next_target = "visualization" if analyst_out.requires_visualization else "insight"

    return {
        "analyst_output": analyst_out,
        "next_agent": next_target,
        "history": history,
    }


def visualization_node(state: DataPilotState) -> Dict[str, Any]:
    """Data Visualization Agent node."""
    query = state.get("query", "")
    dataset_id = state.get("dataset_id")
    df = state.get("df")
    analyst_out = state.get("analyst_output")
    llm = state.get("llm")

    vis_out = run_visualization_agent(
        query=query, dataset_id=dataset_id, df=df, analyst_output=analyst_out, llm=llm
    )
    history = list(state.get("history", []))
    history.append("Visualization agent node generated chart artifact")

    return {
        "vis_output": vis_out,
        "next_agent": "insight",
        "history": history,
    }


def insight_node(state: DataPilotState) -> Dict[str, Any]:
    """Executive Insight Agent node."""
    query = state.get("query", "")
    dataset_id = state.get("dataset_id")
    df = state.get("df")
    analyst_out = state.get("analyst_output")
    vis_out = state.get("vis_output")
    llm = state.get("llm")

    insight_out = run_insight_agent(
        query=query,
        dataset_id=dataset_id,
        df=df,
        analyst_output=analyst_out,
        vis_output=vis_out,
        llm=llm,
    )
    history = list(state.get("history", []))
    history.append("Insight agent node completed executive synthesis")

    return {
        "insight_output": insight_out,
        "next_agent": "end",
        "history": history,
    }


def route_supervisor_edge(state: DataPilotState) -> str:
    """Conditional edge routing based on supervisor output."""
    next_node = state.get("next_agent", "end")
    if next_node in ("profiler", "analyst", "visualization", "insight"):
        return next_node
    return "end"


def route_analyst_edge(state: DataPilotState) -> str:
    """Conditional edge routing based on analyst output requirements."""
    next_node = state.get("next_agent", "insight")
    if next_node in ("visualization", "insight"):
        return next_node
    return "insight"


def build_datapilot_graph():
    """Builds and compiles the DataPilot Multi-Agent StateGraph."""
    graph = StateGraph(DataPilotState)

    # Add Nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("profiler", profiler_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("visualization", visualization_node)
    graph.add_node("insight", insight_node)

    # Set Entry Point
    graph.add_edge(START, "supervisor")

    # Add Conditional Edges from Supervisor
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor_edge,
        {
            "profiler": "profiler",
            "analyst": "analyst",
            "visualization": "visualization",
            "insight": "insight",
            "end": END,
        },
    )

    # Add Edges from Worker Nodes
    graph.add_edge("profiler", END)
    graph.add_conditional_edges(
        "analyst",
        route_analyst_edge,
        {
            "visualization": "visualization",
            "insight": "insight",
        },
    )
    graph.add_edge("visualization", "insight")
    graph.add_edge("insight", END)

    return graph.compile()
