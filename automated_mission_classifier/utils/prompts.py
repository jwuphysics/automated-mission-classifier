"""Prompt management utilities."""

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


PROMPT_FILES = {
    'science_system': 'science_system.txt',
    'science_user': 'science_user.txt',
    'rerank_science_query': 'rerank_science_query.txt',
}


def load_prompts(prompts_dir: Path) -> Dict[str, str]:
    """Loads system and user prompt templates from files."""
    loaded_prompts = {}
    logger.info(f"Loading prompts from directory: {prompts_dir}")
    
    if not prompts_dir.is_dir():
        logger.warning(f"Prompts directory not found: {prompts_dir}. Creating it.")
        try:
            prompts_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create prompts directory {prompts_dir}: {e}")
            raise 

    for key, filename in PROMPT_FILES.items():
        filepath = prompts_dir / filename
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_prompts[key] = f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {filepath}. Please create it.")
            raise FileNotFoundError(f"Essential prompt file missing: {filepath}") 
        except Exception as e:
            logger.error(f"Failed to load prompt file {filepath}: {e}")
            raise 

    return loaded_prompts