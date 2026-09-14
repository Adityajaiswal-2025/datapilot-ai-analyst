import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db_session
from app.services.dataset_service import DatasetService
from app.schemas.agents import (
    AgentQueryRequest,
    AgentQueryResponse,
)
from app.data.metadata import get_registered_dataset
from app.memory.storage import default_session_storage
from app.memory.manager import sync_session_turn
from app.agents.graph import build_datapilot_graph

logger = logging.getLogger("datapilot.api.routes.agents")

router = APIRouter()


@router.post(
    "/query",
    response_model=AgentQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute multi-agent analytical workflow",
    description="Processes natural language queries via Supervisor StateGraph routing across specialized profiler, analyst, visualization, and insight agents.",
)
async def execute_agent_query(
    req: AgentQueryRequest,
    db: AsyncSession = Depends(get_db_session),
) -> AgentQueryResponse:
    """Executes the multi-agent DataPilot workflow for a user query."""
    if not req.query or not req.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    # Authoritative dataset resolution & RAM cache rehydration via DatasetService
    if req.dataset_id:
        service = DatasetService(db)
        await service.get_dataframe(req.dataset_id)

    # Get or create conversation session
    session = default_session_storage.get_or_create_session(
        session_id=req.session_id, dataset_id=req.dataset_id
    )

    # Build and invoke LangGraph StateGraph workflow
    graph = build_datapilot_graph()
    initial_state: Dict[str, Any] = {
        "query": req.query,
        "session_id": session.session_id,
        "dataset_id": req.dataset_id,
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        logger.error(f"StateGraph workflow execution error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-agent workflow execution failed: {str(e)}",
        )

    # Extract outputs
    rewritten_q = final_state.get("query", req.query)
    sup_out = final_state.get("supervisor_output")
    prof_out = final_state.get("profiler_output")
    analyst_out = final_state.get("analyst_output")
    vis_out = final_state.get("vis_output")
    insight_out = final_state.get("insight_output")
    next_agent = sup_out.next_agent if sup_out else final_state.get("next_agent", "end")
    history_trace = final_state.get("history", [])
    errors_trace = final_state.get("errors", [])

    # Synthesize assistant response for session history
    if insight_out:
        assistant_resp = insight_out.executive_summary
    elif analyst_out:
        assistant_resp = analyst_out.findings_summary
    elif prof_out:
        assistant_resp = prof_out.dataset_summary
    else:
        assistant_resp = f"Workflow completed with target '{next_agent}'."

    # Update session memory turn
    sync_session_turn(
        session_id=session.session_id,
        user_query=req.query,
        rewritten_query=rewritten_q,
        assistant_response=assistant_resp,
        dataset_id=req.dataset_id,
        metadata={
            "next_agent": next_agent,
            "has_chart": bool(vis_out and vis_out.is_useful),
        },
    )

    return AgentQueryResponse(
        session_id=session.session_id,
        query=req.query,
        rewritten_query=rewritten_q if rewritten_q != req.query else None,
        next_agent=next_agent,
        supervisor=sup_out,
        profiler=prof_out,
        analyst=analyst_out,
        visualization=vis_out,
        insight=insight_out,
        history=history_trace,
        errors=errors_trace,
    )


@router.get(
    "/sessions/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Retrieve session conversation history",
    description="Fetches stored chat messages and metadata for a given conversation session ID.",
)
def get_session_details(session_id: str) -> Dict[str, Any]:
    """Retrieves session details and message history."""
    session = default_session_storage.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID '{session_id}' not found.",
        )
    return session.model_dump()


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete conversation session",
    description="Deletes persistent session file and clears memory cache for a given session ID.",
)
def delete_session(session_id: str) -> Dict[str, str]:
    """Deletes a conversation session."""
    session = default_session_storage.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID '{session_id}' not found.",
        )
    default_session_storage.delete_session(session_id)
    return {"message": f"Session '{session_id}' deleted successfully."}
