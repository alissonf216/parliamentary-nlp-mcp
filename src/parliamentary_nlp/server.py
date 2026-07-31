"""FastMCP / MCPServer exposing parliamentary discourse safety auditing.

Compatible with the official ``mcp`` Python SDK 1.x (``FastMCP``) and 2.x
(``MCPServer``, the FastMCP successor).

Run via::

    parliamentary-nlp-mcp
    # or
    python -m parliamentary_nlp.server

Inspect interactively with the MCP Inspector::

    npx @modelcontextprotocol/inspector parliamentary-nlp-mcp
"""

from __future__ import annotations

from typing import Any

try:
    # mcp SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server
except ImportError:  # pragma: no cover - exercised on mcp>=2
    # mcp SDK 2.x — FastMCP was renamed to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server

from parliamentary_nlp.model import AuditResult, get_model

mcp = _Server("Parliamentary-NLP-Auditor")


@mcp.tool()
def audit_parliamentary_speech(text: str) -> dict[str, Any]:
    """Audit Portuguese parliamentary or political speech for offensive content.

    Use this tool whenever a user asks you to analyse, moderate, classify, or
    safety-check political / legislative discourse in Portuguese (PT-BR),
    including floor speeches, committee interventions, social-media posts by
    elected officials, and campaign rhetoric.

    The underlying BERTimbau-based classifier estimates:
    - Predominant category (neutral vs. offense / hate-speech tiers)
    - Softmax confidence for the top class
    - Shannon-entropy uncertainty; high entropy (``> 0.60``) sets
      ``requires_human_review=True`` so borderline cases can be escalated

    Args:
        text: Raw Portuguese utterance or transcript excerpt to audit.

    Returns:
        Structured audit dictionary with ``classification``, ``confidence``,
        ``entropy_uncertainty``, ``class_probabilities``, and
        ``requires_human_review``.
    """
    result: AuditResult = get_model().predict(text)
    return dict(result)


def main() -> None:
    """Entry point for the ``parliamentary-nlp-mcp`` console script."""
    mcp.run()


if __name__ == "__main__":
    main()
