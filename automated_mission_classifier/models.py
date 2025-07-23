"""Data models for mission science analysis."""

from pydantic import BaseModel, Field


class MissionScienceLabelerModel(BaseModel):
    """Model for mission science classification results."""
    quotes: list[str] = Field(..., description="A list of quotes supporting the reason, MUST be exact substrings from the provided excerpts.")
    science: float = Field(..., description="Whether the paper contains mission science, scored between 0 and 1")
    reason: str = Field(..., description="Justification for the given 'science' score based ONLY on the provided excerpts")


# Backward compatibility alias
JWSTScienceLabelerModel = MissionScienceLabelerModel