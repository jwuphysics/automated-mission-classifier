"""Command-line interface for Automated Mission Classifier."""

import argparse
import logging
import re
import sys
from pathlib import Path

from .analyzer import AutomatedMissionClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Classify papers as science or non-science for specified MAST missions using LLM analysis of full-text content.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--mission",
        required=True,
        help="Mission name (e.g., TESS, GALEX, PANSTARRS) for classification"
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("./data/combined_dataset_2025_03_25.json"),
        help="Path to JSON data file containing paper records"
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--bibcode", 
        help="Specific bibcode to process for single paper analysis."
    )
    mode_group.add_argument(
        "--batch-mode", 
        help="Batch processing mode: 'all' for all papers, path to file with bibcodes (one per line), or comma-separated list of bibcodes."
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./"),
        help="Project directory where papers/, texts/, and results/ subdirectories will be created. Defaults to current directory."
    )
    parser.add_argument(
        "--prompts-dir", "-p",
        type=Path,
        default=Path("./prompts"), 
        help="Directory containing LLM prompt template files (e.g., science_system.txt)."
    )
    parser.add_argument(
        "--science-threshold",
        type=float,
        default=0.5, 
        help="Threshold for classifying papers as mission science (0-1)"
    )
    parser.add_argument(
        "--reranker-threshold",
        type=float,
        default=0.05,
        help="Minimum reranker score for the top snippet to proceed with LLM analysis. Scores below this threshold will skip the LLM call (range 0-1)."
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force reprocessing of downloaded/analyzed papers"
    )
    parser.add_argument(
        "--top-k-snippets",
        type=int,
        default=5, 
        help="Number of top reranked snippets to send to the LLM"
    )
    parser.add_argument(
        "--context-sentences",
        type=int,
        default=3, 
        help="Number of sentences before and after a keyword sentence to include in a snippet"
    )
    parser.add_argument(
        "--cohere-reranker-model",
        default="rerank-v3.5", 
        help="Cohere reranker model name (when using legacy reranking)"
    )
    parser.add_argument(
        "--gpt-model",
        default="gpt-4.1-mini-2025-04-14", 
        help="GPT scoring model for mission science classification"
    )
    parser.add_argument(
        "--no-gpt-reranker",
        action="store_true",
        help="Use the legacy Cohere reranker instead of the default GPT-4.1-nano reranker"
    )
    parser.add_argument(
        "--limit-papers",
        type=int,
        help="Limit processing to the first N papers (useful for testing). Only applies to batch mode."
    )
    parser.add_argument("--openai-key", help="OpenAI API key (uses OPENAI_API_KEY env var if not provided)")
    parser.add_argument("--cohere-key", help="Cohere API key (uses COHERE_API_KEY env var if not provided; reranking skipped if missing)")

    args = parser.parse_args()

    # Validate thresholds
    if not 0 <= args.science_threshold <= 1:
        parser.error("Science threshold must be between 0 and 1")
    
    # Validate limit-papers
    if args.limit_papers is not None and args.limit_papers < 1:
        parser.error("--limit-papers must be a positive integer")

    # Validate data file exists
    if not args.data_file.exists():
        parser.error(f"Data file not found: {args.data_file}")
        
    # Process batch mode argument
    batch_mode_processed = None
    if args.batch_mode:
        if args.batch_mode.lower() == 'all':
            batch_mode_processed = 'all'
        elif Path(args.batch_mode).exists():
            batch_mode_processed = args.batch_mode
        elif ',' in args.batch_mode:
            batch_mode_processed = [b.strip() for b in args.batch_mode.split(',') if b.strip()]
        else:
            parser.error("--batch-mode must be 'all', a file path, or comma-separated bibcodes")

    # Create output/prompts directory if it doesn't exist
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.prompts_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create necessary directories ({args.output_dir}, {args.prompts_dir}): {e}")
        sys.exit(1)

    try:
        analyzer = AutomatedMissionClassifier(
            output_dir=args.output_dir,
            data_file=args.data_file,
            mission=args.mission,
            bibcode=args.bibcode,
            batch_mode=batch_mode_processed,
            prompts_dir=args.prompts_dir, 
            science_threshold=args.science_threshold,
            reranker_threshold=args.reranker_threshold,
            openai_key=args.openai_key,
            cohere_key=args.cohere_key,
            gpt_model=args.gpt_model,
            cohere_reranker_model=args.cohere_reranker_model,
            top_k_snippets=args.top_k_snippets,
            context_sentences=args.context_sentences,
            reprocess=args.reprocess,
            use_gpt_reranker=not args.no_gpt_reranker,
            limit_papers=args.limit_papers,
        )

        if analyzer.run_mode == "batch":
            analyzer.run_batch() 
        elif analyzer.run_mode == "single":
            analyzer.process_single_paper(args.bibcode) 

    except ValueError as e:
        logger.error(f"Initialization Error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"Setup Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during execution: {e}")
        logger.exception("Traceback:")
        sys.exit(1)

    logger.info("Script finished successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
