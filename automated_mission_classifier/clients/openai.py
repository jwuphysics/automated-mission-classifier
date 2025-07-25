"""OpenAI API client wrapper."""

import json
import logging
from typing import Optional, Dict, Any, Type

from openai import OpenAI, BadRequestError
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Wrapper for OpenAI API interactions."""
    
    def __init__(self, api_key: str, model: str = 'gpt-4.1-mini-2025-04-14'):
        self.client = OpenAI(api_key=api_key, max_retries=2)
        self.model = model
        
    def call_parse(self, system_prompt: str, user_prompt: str, 
                   response_model: Type[BaseModel]) -> Optional[Dict[str, Any]]:
        """Helper function to call OpenAI parse API with error handling."""
        try:
            if not system_prompt or not user_prompt:
                logger.error("System or User prompt is empty. Cannot call OpenAI.")
                return {"error": "empty_prompt", "message": "System or User prompt was empty."}

            result = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=response_model,
                timeout=60 
            )

            parsed_object = result.choices[0].message.parsed
            if parsed_object:
                if isinstance(parsed_object, response_model):
                    return parsed_object.model_dump()
                else:
                    logger.error(f"OpenAI response parsed, but yielded unexpected type: {type(parsed_object)}. Expected {response_model}.")
                    logger.debug(f"Raw parsed object: {parsed_object}")
                    return {"error": "parse_type_mismatch", "message": f"Parsed object type mismatch: got {type(parsed_object)}"}
            else:
                logger.error("OpenAI response parsed, but no Pydantic object found in expected location.")
                logger.debug(f"Full OpenAI response object: {result}")
                try:
                    raw_content = result.choices[0].message.content
                    if isinstance(raw_content, str):
                        parsed_json = json.loads(raw_content)
                        validated_model = response_model.model_validate(parsed_json)
                        logger.warning("Used manual JSON parsing/validation fallback.")
                        return validated_model.model_dump()
                    else:
                        logger.error("Raw message content is not a string for manual parsing.")
                        return None
                except (json.JSONDecodeError, ValidationError, Exception) as manual_parse_err:
                    logger.error(f"Manual parsing/validation of OpenAI response failed: {manual_parse_err}")
                    return None

        except BadRequestError as e:
            logger.warning(f"OpenAI API BadRequestError: {e}")
            if "context length" in str(e).lower():
                logger.warning(f"Snippets might still exceed token limit: {e}")
                return {"error": "token_limit", "message": str(e)}
            else:
                return {"error": "bad_request", "message": str(e)}

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return {"error": "api_error", "message": str(e)}

        return None
    
    def call_separated_analysis(self, system_prompt: str, user_prompt: str, mission: str) -> Optional[Dict[str, Any]]:
        """Completely separated analysis: reasoning first, then scoring based on reasoning."""
        from ..models import MissionScienceReasoningModel, MissionScienceScoringModel
        
        try:
            # Step 1: Get reasoning and quotes only
            reasoning_result = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=MissionScienceReasoningModel,
                timeout=60
            )
            
            reasoning_data = reasoning_result.choices[0].message.parsed
            if not reasoning_data:
                logger.error("Failed to get reasoning from first step")
                return None
                
            reasoning_dict = reasoning_data.model_dump()
            
            # Step 2: Score based on the reasoning
            scoring_prompt = f"""Based on this analysis of {mission} mission science evidence:

REASONING: {reasoning_dict['reason']}

What score from 0.0 to 1.0 does this reasoning support for **{mission}** mission science? Your score must be consistent with the reasoning above. Pay close attention to any explicit score mentioned in the reasoning."""
            
            scoring_result = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "user", "content": scoring_prompt}
                ],
                response_format=MissionScienceScoringModel,
                timeout=60
            )
            
            scoring_data = scoring_result.choices[0].message.parsed
            if not scoring_data:
                logger.error("Failed to get score from second step")
                return None
                
            # Combine results
            return {
                "quotes": reasoning_dict["quotes"],
                "reason": reasoning_dict["reason"], 
                "science": scoring_data.model_dump()["science"]
            }
                
        except Exception as e:
            logger.error(f"Separated analysis failed: {e}")
            return None
