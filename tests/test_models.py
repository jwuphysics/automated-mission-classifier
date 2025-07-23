"""Tests for automated_mission_classifier.models module."""

import pytest
from pydantic import ValidationError

from automated_mission_classifier.models import MissionScienceLabelerModel


class TestMissionScienceLabelerModel:
    """Test cases for MissionScienceLabelerModel."""
    
    def test_valid_model_creation(self):
        """Test creating a valid MissionScienceLabelerModel instance."""
        data = {
            "quotes": ["This paper uses TESS observations", "We analyze TESS photometry"],
            "science": 0.9,
            "reason": "Paper presents clear TESS science content"
        }
        model = MissionScienceLabelerModel(**data)
        
        assert model.quotes == data["quotes"]
        assert model.science == data["science"]
        assert model.reason == data["reason"]
    
    def test_score_validation_range(self):
        """Test that science score validation works correctly."""
        # Valid scores (0.0 to 1.0)
        valid_data = {
            "quotes": ["Sample quote"],
            "science": 0.5,
            "reason": "Sample reason"
        }
        model = MissionScienceLabelerModel(**valid_data)
        assert model.science == 0.5
        
        # Edge cases
        edge_cases = [0.0, 1.0]
        for score in edge_cases:
            valid_data["science"] = score
            model = MissionScienceLabelerModel(**valid_data)
            assert model.science == score
    
    def test_empty_quotes_list(self):
        """Test model creation with empty quotes list."""
        data = {
            "quotes": [],
            "science": 0.1,
            "reason": "No supporting quotes found"
        }
        model = MissionScienceLabelerModel(**data)
        assert model.quotes == []
    
    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        incomplete_data = {"quotes": ["Sample quote"]}
        
        with pytest.raises(ValidationError) as exc_info:
            MissionScienceLabelerModel(**incomplete_data)
        
        errors = exc_info.value.errors()
        missing_fields = {error["loc"][0] for error in errors}
        assert "science" in missing_fields
        assert "reason" in missing_fields
    
    def test_invalid_score_type(self):
        """Test that invalid score types raise ValidationError."""
        invalid_data = {
            "quotes": ["Sample quote"],
            "science": "invalid",
            "reason": "Sample reason"
        }
        
        with pytest.raises(ValidationError):
            MissionScienceLabelerModel(**invalid_data)
    
    def test_backward_compatibility_jwstscience(self):
        """Test that the model still works with the old jwstscience field."""
        data = {
            "quotes": ["This paper uses JWST observations"],
            "jwstscience": 0.8,
            "reason": "Paper presents JWST science content"
        }
        model = MissionScienceLabelerModel(**data)
        
        assert model.science == 0.8
        assert model.jwstscience == 0.8
    
    def test_new_science_field(self):
        """Test that the model works with the new science field."""
        data = {
            "quotes": ["This paper uses TESS observations"],
            "science": 0.7,
            "reason": "Paper presents TESS science content"
        }
        model = MissionScienceLabelerModel(**data)
        
        assert model.science == 0.7
        assert model.jwstscience == 0.7