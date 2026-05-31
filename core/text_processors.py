"""
Text processing utilities for parsing, cleaning, and extracting information.
"""
import re
import json
import ast


def parse_possible_dict(text: str):
    """Try to parse text as JSON or Python literal."""
    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def extract_json_from_text(text: str) -> dict:
    """Extract JSON from text, even if wrapped in markdown code blocks."""
    if not text:
        raise ValueError("Empty LLM response.")

    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```JSON", "")
    cleaned = cleaned.replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Could not parse JSON from LLM response: {text}")


def clean_agent_text(text: str) -> str:
    """Remove raw tool call dictionaries and internal tags from text."""
    if not text:
        return ""

    clean_lines = []

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("{'type': 'function_call'"):
            continue
        if stripped.startswith('{"type": "function_call"'):
            continue
        if stripped.startswith("{'type': 'function_call_output'"):
            continue
        if stripped.startswith('{"type": "function_call_output"'):
            continue
        if stripped.startswith("<name>") and stripped.endswith("</name>"):
            continue

        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def extract_tool_traces_from_text(text: str) -> list:
    """Extract tool call traces from text for Langfuse metadata."""
    traces = []

    if not text:
        return traces

    for line in text.splitlines():
        stripped = line.strip()

        is_tool_line = (
            stripped.startswith("{'type': 'function_call'")
            or stripped.startswith('{"type": "function_call"')
            or stripped.startswith("{'type': 'function_call_output'")
            or stripped.startswith('{"type": "function_call_output"')
        )

        if is_tool_line:
            parsed = parse_possible_dict(stripped)
            traces.append(parsed if parsed else stripped)

    return traces


def extract_text_from_content_blocks(content_blocks) -> str:
    """Extract final text blocks and ignore reasoning blocks."""
    text_parts = []

    if not isinstance(content_blocks, list):
        return str(content_blocks).strip()

    for item in content_blocks:
        if not isinstance(item, dict):
            continue

        block_type = item.get("type")

        # Ignore reasoning blocks
        if block_type == "reasoning":
            continue

        if block_type in ["text", "output_text"]:
            text_value = item.get("text") or item.get("content") or ""
            if text_value:
                text_parts.append(str(text_value))

        elif "text" in item and block_type != "reasoning":
            text_parts.append(str(item["text"]))

        elif "content" in item and block_type != "reasoning":
            text_parts.append(str(item["content"]))

    final_text = "\n".join(text_parts).strip()

    if final_text:
        return final_text

    return "No final message was returned by the model. Please retry with a shorter instruction."


def extract_llm_text(result: dict) -> str:
    """Extract LLM text handling various content formats."""
    try:
        content = result["choices"][0]["message"]["content"]

        if isinstance(content, str):
            cleaned = content.strip()

            if cleaned.startswith("[") and cleaned.endswith("]"):
                try:
                    parsed = ast.literal_eval(cleaned)
                    return extract_text_from_content_blocks(parsed)
                except Exception:
                    return cleaned

            return cleaned

        if isinstance(content, list):
            return extract_text_from_content_blocks(content)

        return str(content).strip()

    except Exception as e:
        return f"Unable to parse LLM response: {e}"


def contains_kannada(text: str) -> bool:
    """Check if text contains Kannada script characters."""
    return bool(re.search(r"[\u0C80-\u0CFF]", text or ""))


def contains_devanagari(text: str) -> bool:
    """Check if text contains Devanagari script (Hindi) characters."""
    return bool(re.search(r"[\u0900-\u097F]", text or ""))
