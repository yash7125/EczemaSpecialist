import json
import os
import re
from typing import Any, TypedDict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from orchestrator_agent.prompt import ORCHESTRATOR_PROMPT


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(BASE_DIR, "knowledge_base_H&R.json")


def consult_specialist(
    image_path: str,
    prompt: str = "Analyze the morphology of the lesion in this image.",
) -> str:
    """Consult the Dermatology Specialist Agent via the A2A protocol."""
    import base64
    import requests
    import uuid

    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "id": str(uuid.uuid4()),
            "params": {
                "message": {
                    "messageId": f"msg_{uuid.uuid4().hex[:8]}",
                    "role": "user",
                    "kind": "message",
                    "parts": [
                        {"kind": "text", "text": prompt},
                        {
                            "kind": "file",
                            "file": {
                                "bytes": encoded_string,
                                "mimeType": "image/jpeg",
                            },
                        },
                    ],
                }
            },
        }

        response = requests.post("http://localhost:8001/", json=payload, timeout=120.0)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            error_info = data["error"]
            error_msg = error_info.get("message", str(error_info))
            return f"Specialist Agent Error: {error_msg[:200]}"

        result = data.get("result", {})
        parts = result.get("parts") or result.get("message", {}).get("parts", [])
        if not parts:
            parts = result.get("status", {}).get("message", {}).get("parts", [])
        if not parts:
            for artifact in result.get("artifacts", []):
                parts.extend(artifact.get("parts", []))

        for part in parts:
            if part.get("text"):
                return part["text"]
            if part.get("kind") == "text" and "text" in part:
                return part["text"]

        return "Specialist analysis completed, but no text was returned."
    except FileNotFoundError:
        return f"Error: Image file not found at path '{image_path}'."
    except Exception as exc:
        return f"Error communicating with Specialist Agent: {str(exc)}"


class State(TypedDict):
    messages: list[BaseMessage]
    patient_facts: dict[str, dict[str, Any]]
    image_path: str
    specialist_findings: str
    reference_data: dict[str, str]
    evidence_ledger: list[dict[str, str]]
    criteria_status: dict[str, str]
    open_questions: list[str]
    interview_status: str
    asked_topics: list[str]
    topic_attempts: dict[str, int]
    previous_criteria_status: dict[str, str]
    stalled_turns: int
    stop_reason: str
    turn_count: int


QUESTION_BANK = {
    "pruritus": "How intense is the itching, and does it keep coming back or disturb sleep?",
    "chronic_relapsing": "Has this rash been recurring for months or years, or is this the first short episode?",
    "typical_distribution": "Which body areas are involved most often, such as elbow folds, behind the knees, neck, face, or hands?",
    "atopy_history": "Do you or close family members have eczema, asthma, or allergic rhinitis?",
    "xerosis": "Do you usually have generally dry or sensitive skin even between flares?",
    "childhood_onset": "Did these skin problems begin in childhood or only later in life?",
}

PRIORITY_ORDER = [
    "pruritus",
    "chronic_relapsing",
    "typical_distribution",
    "atopy_history",
    "xerosis",
    "childhood_onset",
]

MAX_TOPIC_ATTEMPTS = 2

def _extract_image_path(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        content = str(message.content)
        match = re.search(r"Image provided at:\s*(.+)", content)
        if match:
            return match.group(1).strip()
    return ""


def _load_kb() -> dict[str, Any]:
    with open(KB_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)

def _normalize_text(messages: list[BaseMessage], message_type: type[BaseMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message, message_type):
            parts.append(str(message.content))
    return "\n".join(parts)

def _empty_patient_facts() -> dict[str, dict[str, Any]]:
    kb = _load_kb()
    descriptions = {
        "pruritus": kb["major_criteria"]["pruritus"],
        "chronic_relapsing": kb["major_criteria"]["chronic_history"],
        "typical_distribution": kb["major_criteria"]["distribution"],
        "atopy_history": kb["major_criteria"]["atopy_history"],
        "xerosis": kb["minor_criteria"]["xerosis"],
        "childhood_onset": kb["minor_criteria"]["early_age_onset"],
    }
    return {
        criterion: {
            "status": "uncertain",
            "positive_similarity": 0.0,
            "negative_similarity": 0.0,
            "description": description,
        }
        for criterion, description in descriptions.items()
    }


async def _extract_patient_facts(tool_map: dict[str, Any], messages: list[BaseMessage]) -> dict[str, dict[str, Any]]:
    patient_history = _normalize_text(messages, HumanMessage)
    if not patient_history.strip():
        return _empty_patient_facts()

    score_tool = tool_map.get("score_patient_history")
    if score_tool is None:
        return _empty_patient_facts()

    try:
        result = await _invoke_tool(score_tool, {"patient_history": patient_history})
        parsed = json.loads(result)
    except Exception:
        return _empty_patient_facts()

    fallback = _empty_patient_facts()
    for criterion, default_value in fallback.items():
        value = parsed.get(criterion)
        if isinstance(value, dict):
            fallback[criterion] = {
                "status": value.get("status", default_value["status"]),
                "positive_similarity": value.get("positive_similarity", default_value["positive_similarity"]),
                "negative_similarity": value.get("negative_similarity", default_value["negative_similarity"]),
                "description": value.get("description", default_value["description"]),
            }
    return fallback


def _empty_specialist_support() -> dict[str, dict[str, Any]]:
    return {
        "morphology_support": {
            "status": "uncertain",
            "positive_similarity": 0.0,
            "negative_similarity": 0.0,
            "description": "Typical atopic dermatitis morphology support from specialist findings.",
        },
        "distribution_support": {
            "status": "uncertain",
            "positive_similarity": 0.0,
            "negative_similarity": 0.0,
            "description": "Typical atopic dermatitis distribution support from specialist findings.",
        },
    }


async def _summarize_specialist_support(tool_map: dict[str, Any], specialist_findings: str) -> dict[str, dict[str, Any]]:
    if not specialist_findings or specialist_findings.lower().startswith(("error", "specialist agent error")):
        return _empty_specialist_support()

    score_tool = tool_map.get("score_specialist_findings")
    if score_tool is None:
        return _empty_specialist_support()

    try:
        result = await _invoke_tool(score_tool, {"specialist_findings": specialist_findings})
        parsed = json.loads(result)
    except Exception:
        return _empty_specialist_support()

    fallback = _empty_specialist_support()
    for criterion, default_value in fallback.items():
        value = parsed.get(criterion)
        if isinstance(value, dict):
            fallback[criterion] = {
                "status": value.get("status", default_value["status"]),
                "positive_similarity": value.get("positive_similarity", default_value["positive_similarity"]),
                "negative_similarity": value.get("negative_similarity", default_value["negative_similarity"]),
                "description": value.get("description", default_value["description"]),
            }
    return fallback


def _build_evidence_ledger(
    patient_facts: dict[str, dict[str, Any]],
    specialist_support: dict[str, dict[str, Any]],
    reference_data: dict[str, str],
) -> tuple[list[dict[str, str]], dict[str, str], list[str]]:
    criteria_status = {
        "pruritus": patient_facts["pruritus"]["status"],
        "chronic_relapsing": patient_facts["chronic_relapsing"]["status"],
        "typical_distribution": "met"
        if (
            patient_facts["typical_distribution"]["status"] == "met"
            or specialist_support["distribution_support"]["status"] == "met"
        )
        else patient_facts["typical_distribution"]["status"],
        "atopy_history": patient_facts["atopy_history"]["status"],
        "xerosis": patient_facts["xerosis"]["status"],
        "childhood_onset": patient_facts["childhood_onset"]["status"],
        "morphology_support": specialist_support["morphology_support"]["status"],
    }

    source_has_kb = "met" if reference_data else "uncertain"
    ledger = [
        {
            "criterion": "pruritus",
            "status": criteria_status["pruritus"],
            "source": "patient",
            "evidence": (
                "Semantic match to pruritus hypotheses. "
                f"positive={patient_facts['pruritus']['positive_similarity']}, "
                f"negative={patient_facts['pruritus']['negative_similarity']}"
            ),
        },
        {
            "criterion": "typical_distribution",
            "status": criteria_status["typical_distribution"],
            "source": "patient/specialist",
            "evidence": (
                "Semantic match to typical AD distribution hypotheses plus image findings. "
                f"history_positive={patient_facts['typical_distribution']['positive_similarity']}, "
                f"history_negative={patient_facts['typical_distribution']['negative_similarity']}"
            ),
        },
        {
            "criterion": "chronic_relapsing_course",
            "status": criteria_status["chronic_relapsing"],
            "source": "patient",
            "evidence": (
                "Semantic match to chronic relapsing course hypotheses. "
                f"positive={patient_facts['chronic_relapsing']['positive_similarity']}, "
                f"negative={patient_facts['chronic_relapsing']['negative_similarity']}"
            ),
        },
        {
            "criterion": "personal_or_family_atopy",
            "status": criteria_status["atopy_history"],
            "source": "patient",
            "evidence": (
                "Semantic match to atopy history hypotheses. "
                f"positive={patient_facts['atopy_history']['positive_similarity']}, "
                f"negative={patient_facts['atopy_history']['negative_similarity']}"
            ),
        },
        {
            "criterion": "xerosis",
            "status": criteria_status["xerosis"],
            "source": "patient",
            "evidence": (
                "Semantic match to xerosis hypotheses. "
                f"positive={patient_facts['xerosis']['positive_similarity']}, "
                f"negative={patient_facts['xerosis']['negative_similarity']}"
            ),
        },
        {
            "criterion": "childhood_onset",
            "status": criteria_status["childhood_onset"],
            "source": "patient",
            "evidence": (
                "Semantic match to early age onset hypotheses. "
                f"positive={patient_facts['childhood_onset']['positive_similarity']}, "
                f"negative={patient_facts['childhood_onset']['negative_similarity']}"
            ),
        },
        {
            "criterion": "morphology_support",
            "status": criteria_status["morphology_support"],
            "source": "specialist",
            "evidence": (
                "Semantic match to specialist morphology hypotheses. "
                f"positive={specialist_support['morphology_support']['positive_similarity']}, "
                f"negative={specialist_support['morphology_support']['negative_similarity']}"
            ),
        },
        {
            "criterion": "hr_reference_loaded",
            "status": source_has_kb,
            "source": "knowledge base",
            "evidence": "Hanifin & Rajka criteria retrieved from MCP tools.",
        },
    ]

    open_questions = [topic for topic in PRIORITY_ORDER if criteria_status.get(topic, "uncertain") == "uncertain"]
    return ledger, criteria_status, open_questions


def _is_ready_to_conclude(criteria_status: dict[str, str]) -> bool:
    major_keys = ["pruritus", "typical_distribution", "chronic_relapsing", "atopy_history"]
    minor_keys = ["xerosis", "childhood_onset", "morphology_support"]
    major_resolved = sum(criteria_status.get(key) != "uncertain" for key in major_keys)
    major_met = sum(criteria_status.get(key) == "met" for key in major_keys)
    minor_resolved = sum(criteria_status.get(key) != "uncertain" for key in minor_keys)
    return major_resolved == len(major_keys) and major_met >= 2 and minor_resolved >= 2


def _has_meaningful_progress(previous: dict[str, str], current: dict[str, str]) -> bool:
    if not previous:
        return True
    if previous != current:
        return True
    previous_uncertain = sum(status == "uncertain" for status in previous.values())
    current_uncertain = sum(status == "uncertain" for status in current.values())
    return current_uncertain < previous_uncertain


def _clarification_budget_exhausted(open_questions: list[str], topic_attempts: dict[str, int]) -> bool:
    return bool(open_questions) and all(
        topic_attempts.get(topic, 0) >= MAX_TOPIC_ATTEMPTS for topic in open_questions
    )


async def _invoke_tool(tool: Any, payload: Any) -> str:
    result = await tool.ainvoke(payload)
    return str(result)


def create_initial_state(patient_text: str, image_path: str = "") -> State:
    content = f"Patient Input: {patient_text}"
    if image_path:
        content += f"\nImage provided at: {image_path}"
    return {
        "messages": [HumanMessage(content=content)],
        "patient_facts": {},
        "image_path": image_path,
        "specialist_findings": "",
        "reference_data": {},
        "evidence_ledger": [],
        "criteria_status": {},
        "open_questions": [],
        "interview_status": "gathering",
        "asked_topics": [],
        "topic_attempts": {},
        "previous_criteria_status": {},
        "stalled_turns": 0,
        "stop_reason": "",
        "turn_count": 0,
    }


def create_diagnosis_graph(llm, mcp_tools: List):
    """Create a multi-turn LangGraph interview for the eczema orchestrator."""

    tool_map = {tool.name: tool for tool in mcp_tools}

    async def ingest_turn(state: State) -> dict[str, Any]:
        patient_facts = await _extract_patient_facts(tool_map, state["messages"])
        image_path = state.get("image_path") or _extract_image_path(state["messages"])
        turn_count = sum(isinstance(message, HumanMessage) for message in state["messages"])
        return {
            "patient_facts": patient_facts,
            "image_path": image_path,
            "turn_count": turn_count,
        }

    async def retrieve_evidence(state: State) -> dict[str, Any]:
        updates: dict[str, Any] = {}

        image_path = state.get("image_path", "")
        specialist_findings = state.get("specialist_findings", "")
        if image_path and not specialist_findings:
            updates["specialist_findings"] = consult_specialist(image_path)

        reference_data = dict(state.get("reference_data", {}))
        if not reference_data:
            for tool_name in ("get_major_criteria", "get_minor_criteria", "get_grading_rules"):
                tool = tool_map.get(tool_name)
                if tool is not None:
                    reference_data[tool_name] = await _invoke_tool(tool, {})
            updates["reference_data"] = reference_data

        return updates

    async def score_case(state: State) -> dict[str, Any]:
        specialist_support = await _summarize_specialist_support(
            tool_map,
            state.get("specialist_findings", ""),
        )
        ledger, criteria_status, open_questions = _build_evidence_ledger(
            state.get("patient_facts", {}),
            specialist_support,
            state.get("reference_data", {}),
        )
        return {
            "evidence_ledger": ledger,
            "criteria_status": criteria_status,
            "open_questions": open_questions,
        }

    def assess_progress(state: State) -> dict[str, Any]:
        criteria_status = state.get("criteria_status", {})
        previous_criteria_status = state.get("previous_criteria_status", {})
        topic_attempts = dict(state.get("topic_attempts", {}))
        open_questions = state.get("open_questions", [])
        prior_stop_reason = state.get("stop_reason", "")

        made_progress = _has_meaningful_progress(previous_criteria_status, criteria_status)
        stalled_turns = 0 if made_progress else state.get("stalled_turns", 0) + 1

        if prior_stop_reason == "patient_stopped":
            interview_status = "conclude_with_uncertainty"
            stop_reason = prior_stop_reason
        elif _is_ready_to_conclude(criteria_status):
            interview_status = "ready_to_conclude"
            stop_reason = "sufficient_evidence"
        elif _clarification_budget_exhausted(open_questions, topic_attempts):
            interview_status = "conclude_with_uncertainty"
            stop_reason = "clarification_budget_exhausted"
        else:
            interview_status = "needs_more_info"
            stop_reason = ""

        return {
            "previous_criteria_status": criteria_status,
            "stalled_turns": stalled_turns,
            "interview_status": interview_status,
            "stop_reason": stop_reason,
        }

    def respond(state: State) -> dict[str, Any]:
        criteria_status = state.get("criteria_status", {})
        ledger_lines = [
            f"- {item['criterion']}: {item['status']} ({item['source']})"
            for item in state.get("evidence_ledger", [])
        ]
        open_questions = state.get("open_questions", [])
        asked_topics = list(state.get("asked_topics", []))
        topic_attempts = dict(state.get("topic_attempts", {}))
        stop_reason = state.get("stop_reason", "")
        specialist_findings = state.get("specialist_findings", "") or "No specialist findings available."
        terminal_prompt_suffix = (
            f"Specialist findings:\n{specialist_findings}\n\n"
            "Your next message must be a final conclusion only.\n"
            "Do not ask any question.\n"
            "Do not end with a question mark.\n"
        )

        if state.get("interview_status") == "ready_to_conclude":
            prompt = (
                f"{ORCHESTRATOR_PROMPT}\n\n"
                "You now have enough information to provide a concise clinical conclusion.\n"
                "Do not ask another question.\n"
                "Explain the current probability assessment, list which criteria are met or unmet,"
                " and briefly mention any remaining uncertainty.\n\n"
                f"Criteria status: {criteria_status}\n"
                f"Evidence ledger:\n" + "\n".join(ledger_lines) + "\n\n" + terminal_prompt_suffix
            )
        elif state.get("interview_status") == "conclude_with_uncertainty":
            prompt = (
                f"{ORCHESTRATOR_PROMPT}\n\n"
                "You should stop the interview and provide a provisional conclusion based on the available evidence.\n"
                "Do not ask another question.\n"
                "Explain that the interview is concluding with uncertainty, summarize what supports or argues against atopic dermatitis,"
                " and clearly list the most important unresolved criteria.\n"
                "If the patient declined further answers, still use all currently available evidence, including any specialist image findings and retrieved H&R criteria, to provide the best-supported conclusion.\n\n"
                f"Stop reason: {stop_reason}\n"
                f"Criteria status: {criteria_status}\n"
                f"Evidence ledger:\n" + "\n".join(ledger_lines) + "\n\n" + terminal_prompt_suffix
            )
        else:
            available_topics = [
                topic for topic in open_questions if topic_attempts.get(topic, 0) < MAX_TOPIC_ATTEMPTS
            ]
            next_topic = next((topic for topic in available_topics if topic not in asked_topics), "")
            if not next_topic and available_topics:
                next_topic = available_topics[0]
            question_hint = QUESTION_BANK.get(
                next_topic,
                "What other symptom or history detail would help clarify whether this behaves like atopic dermatitis?",
            )
            if next_topic:
                asked_topics.append(next_topic)
                topic_attempts[next_topic] = topic_attempts.get(next_topic, 0) + 1
            image_instruction = ""
            if state.get("image_path"):
                image_instruction = (
                    "\nImage handling rule:\n"
                    "- An image was provided.\n"
                    "- The Dermatology Specialist Agent is the only authorized image analyst.\n"
                    "- Specialist findings have already been retrieved when available.\n"
                    "- Do not say that you are personally analyzing the image.\n"
                    "- Do not say that you will consult the specialist later in this message.\n"
                )
            prompt = (
                f"{ORCHESTRATOR_PROMPT}\n\n"
                "You still need more information before concluding.\n"
                "Respond in two short parts:\n"
                "1. a one or two sentence recap of what is known so far\n"
                "2. exactly one follow-up question\n"
                "Do not give the final conclusion yet.\n\n"
                f"Criteria status: {criteria_status}\n"
                f"Evidence ledger:\n" + "\n".join(ledger_lines) + "\n\n"
                f"Specialist findings:\n{specialist_findings}\n"
                + image_instruction + "\n"
                f"Best next topic to clarify: {next_topic or 'general clarification'}\n"
                f"Suggested question intent: {question_hint}"
            )

        if state.get("interview_status") in {"ready_to_conclude", "conclude_with_uncertainty"}:
            response = llm.invoke([SystemMessage(content=prompt)])
        else:
            response = llm.invoke([SystemMessage(content=prompt), *state["messages"]])
        return {
            "messages": [*state["messages"], response],
            "asked_topics": asked_topics,
            "topic_attempts": topic_attempts,
        }

    workflow = StateGraph(State)
    workflow.add_node("ingest_turn", ingest_turn)
    workflow.add_node("retrieve_evidence", retrieve_evidence)
    workflow.add_node("score_case", score_case)
    workflow.add_node("assess_progress", assess_progress)
    workflow.add_node("respond", respond)

    workflow.add_edge(START, "ingest_turn")
    workflow.add_edge("ingest_turn", "retrieve_evidence")
    workflow.add_edge("retrieve_evidence", "score_case")
    workflow.add_edge("score_case", "assess_progress")
    workflow.add_edge("assess_progress", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile()
