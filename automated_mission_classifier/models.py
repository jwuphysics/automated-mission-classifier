"""Data models for mission science analysis."""

from pydantic import BaseModel, Field


class MissionScienceReasoningModel(BaseModel):
    """Model for mission science reasoning phase."""
    quotes: list[str] = Field(..., description="A list of quotes supporting the analysis, MUST be exact substrings from the provided excerpts.")
    reason: str = Field(..., description="Complete analysis of whether this paper presents new mission science based ONLY on the provided excerpts")


class MissionScienceScoringModel(BaseModel):
    """Model for mission science scoring phase."""
    science: float = Field(..., description="Whether the paper contains mission science, scored between 0 and 1")


class MissionScienceLabelerModel(BaseModel):
    """Model for mission science classification results."""
    quotes: list[str] = Field(..., description="A list of quotes supporting the reason, MUST be exact substrings from the provided excerpts.")
    science: float = Field(..., description="Whether the paper contains mission science, scored between 0 and 1")
    reason: str = Field(..., description="Justification for the given 'science' score based ONLY on the provided excerpts")


# Backward compatibility alias
JWSTScienceLabelerModel = MissionScienceLabelerModel