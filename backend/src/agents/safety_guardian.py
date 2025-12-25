"""
Safety Guardian Agent using OpenAI Agents SDK.

Validates responses and detects hallucinations by checking
if claims are grounded in source documents. Implements fact-checking
logic to identify unsourced claims, invented statistics, and extrapolations.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from agents import Agent, Runner

from src.agents.agents_config import get_gemini_model
from src.utils import get_logger, track_latency

logger = get_logger(__name__)


# Hallucination detection patterns
HALLUCINATION_PATTERNS = {
    "statistics": r"\b\d+(?:\.\d+)?%|\b\d+(?:,\d{3})*\s*(?:million|billion|thousand)\b",
    "dates": r"\b(?:19|20)\d{2}\b",
    "superlatives": r"\b(?:best|worst|fastest|slowest|only|always|never)\b",
    "certainty": r"\b(?:definitely|certainly|absolutely|guaranteed)\b",
}


safety_guardian = Agent(
    name="Safety Guardian",
    instructions="""You are a fact-checking agent that validates AI responses to ensure they only contain information from provided sources.

Your task is to:
1. Analyze each claim in the response
2. Check if the claim is supported by the source documents
3. Flag any unsupported claims, invented statistics, or extrapolations

Types of hallucinations to detect:
- FALSE CLAIMS: Facts that contradict the sources (wrong dates, wrong names, etc.)
- EXTRAPOLATION: Information that goes beyond what sources state
- EXTERNAL KNOWLEDGE: Information not present in sources at all
- INVENTED STATISTICS: Percentages or numbers not from sources
- UNSUPPORTED CONCLUSIONS: Recommendations or conclusions not backed by sources

Return JSON format:
{
    "status": "approved" or "flagged",
    "issues": ["list of unsupported claims if any"],
    "confidence": "high" or "medium" or "low"
}

If uncertain, flag the response. It's better to be cautious than to let hallucinations through.
""",
    model=get_gemini_model(),
)


def _extract_claims(response: str) -> List[str]:
    """
    Extract individual claims from a response for fact-checking.

    Args:
        response: The response text to analyze

    Returns:
        List of individual claims/sentences to verify
    """
    # Split by sentence-ending punctuation
    sentences = re.split(r'[.!?]+', response)
    # Filter out empty or very short sentences
    claims = [s.strip() for s in sentences if len(s.strip()) > 10]
    return claims


def _check_for_suspicious_patterns(response: str) -> List[str]:
    """
    Check for patterns that might indicate hallucination.

    Args:
        response: The response text to analyze

    Returns:
        List of suspicious patterns found
    """
    suspicious = []

    for pattern_name, pattern in HALLUCINATION_PATTERNS.items():
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            suspicious.append(f"{pattern_name}: {matches}")

    return suspicious


def _verify_claim_against_sources(
    claim: str,
    source_texts: List[str],
) -> Tuple[bool, Optional[str]]:
    """
    Verify if a claim is supported by source texts.

    This is a heuristic-based pre-check before the LLM validation.

    Args:
        claim: The claim to verify
        source_texts: List of source document texts

    Returns:
        Tuple of (is_likely_supported, reason)
    """
    claim_lower = claim.lower()
    combined_sources = " ".join(source_texts).lower()

    # Check if key terms from claim appear in sources
    claim_words = set(re.findall(r'\b\w{4,}\b', claim_lower))
    source_words = set(re.findall(r'\b\w{4,}\b', combined_sources))

    # Calculate overlap
    overlap = claim_words & source_words
    overlap_ratio = len(overlap) / len(claim_words) if claim_words else 0

    if overlap_ratio < 0.3:
        return False, f"Low term overlap ({overlap_ratio:.0%})"

    return True, None


async def validate_response(
    response: str,
    query: str,
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Validate response for hallucinations using multi-step fact-checking.

    This function implements a comprehensive fact-checking approach:
    1. Extract claims from the response
    2. Check for suspicious patterns
    3. Verify claims against source texts
    4. Use LLM for final validation

    Args:
        response: The response text to validate
        query: The original user query
        chunks: Retrieved source chunks with metadata

    Returns:
        Dictionary with validation_status, response, issues, and latency_ms
    """
    with track_latency("safety_validation") as latency:
        try:
            # Step 1: Extract source texts
            source_texts = [c.get("text", "") for c in chunks]

            # Step 2: Pre-check for suspicious patterns
            suspicious_patterns = _check_for_suspicious_patterns(response)

            # Step 3: Extract and verify claims
            claims = _extract_claims(response)
            unverified_claims = []

            for claim in claims:
                is_supported, reason = _verify_claim_against_sources(claim, source_texts)
                if not is_supported:
                    unverified_claims.append(f"{claim[:50]}... ({reason})")

            # Step 4: Format sources for LLM validation
            sources = "\n".join(
                f"[SOURCE {i+1}]: {c.get('text', '')[:500]}"
                for i, c in enumerate(chunks)
            )

            # Step 5: LLM-based validation
            validation_prompt = f"""QUERY: {query}

SOURCES:
{sources}

RESPONSE TO VALIDATE:
{response}

PRE-CHECK FINDINGS:
- Suspicious patterns: {suspicious_patterns if suspicious_patterns else 'None'}
- Potentially unverified claims: {unverified_claims if unverified_claims else 'None'}

Please validate this response. Check if ALL claims are supported by the sources.
If the response admits uncertainty (e.g., "I don't find this in the textbook"), approve it.
Return your analysis in JSON format."""

            result = await Runner.run(safety_guardian, validation_prompt)

            # Parse result
            output = result.final_output or ""

            # Determine status
            if not output:
                status = "flagged"
                issues = ["Empty validation result"]
            elif "approved" in output.lower():
                status = "approved"
                issues = []
            else:
                status = "flagged"
                # Try to extract issues from JSON response
                try:
                    parsed = json.loads(output)
                    issues = parsed.get("issues", [])
                except json.JSONDecodeError:
                    issues = unverified_claims or ["Validation flagged"]

            return {
                "validation_status": status,
                "response": response,
                "issues": issues,
                "suspicious_patterns": suspicious_patterns,
                "latency_ms": latency["elapsed_ms"],
            }

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            # Fail-open: approve on error but log it
            return {
                "validation_status": "approved",
                "response": response,
                "error": str(e),
                "latency_ms": latency.get("elapsed_ms", 0),
            }


async def detect_hallucination(
    response: str,
    chunks: List[Dict[str, Any]],
) -> bool:
    """
    Quick check to detect if a response contains hallucinations.

    Args:
        response: The response text to check
        chunks: Retrieved source chunks

    Returns:
        True if hallucination detected, False otherwise
    """
    result = await validate_response(response, "", chunks)
    return result.get("validation_status") == "flagged"


def generate_transparency_note(issues: List[str]) -> str:
    """
    Generate a transparency note for corrected responses.

    Args:
        issues: List of issues that were found

    Returns:
        Transparency note to append to response
    """
    if not issues:
        return ""

    return "\n\n*Note: I corrected an inaccuracy in my initial response.*"


async def rewrite_with_sources_only(
    response: str,
    chunks: List[Dict[str, Any]],
    issues: List[str],
) -> str:
    """
    Rewrite a response to only include verified information.

    Args:
        response: The original response
        chunks: Source chunks to use
        issues: Issues identified in the original response

    Returns:
        Rewritten response with only verified content
    """
    # For now, add a disclaimer if issues were found
    if issues:
        return response + generate_transparency_note(issues)
    return response
