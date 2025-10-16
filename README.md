# Automated Mission Classifier

This tool classifies whether astronomical papers are relevant to specific telescopes or missions (TESS, JWST, GALEX, PANSTARRS, etc.) using LLM analysis of full-body text.

## Quick Start
There are two main ways to use the Automated Mission Classifier:

**Single paper mode**: `amc --mission MISSION_NAME --bibcode BIBCODE`. For example:

```bash
amc --mission TESS --bibcode 2020MNRAS.491.2982E
```

**Batch mode**: `amc --mission MISSION_NAME --batch-mode MODE`. For example:

```bash
# Process specific bibcodes from a file
amc --mission JWST --batch-mode bibcodes.txt

# Process comma-separated bibcodes
amc --mission TESS --batch-mode "2020MNRAS.491.2982E,2020MNRAS.495.2844S"

# Process all papers in the dataset
amc --mission GALEX --batch-mode all
```

## Installation

We recommend using version Python 3.10 or higher, and using a virtual environment. This has so far only been tested on macOS and Linux.

To install from the source, first copy the repository to your computer
```bash
git clone git@github.com:jwuphysics/automated-mission-classifier.git

cd automated-mission-classifier
```

Then, create a virtual environment. An easy way to do this is using `uv`:
```bash
uv venv && source .venv/bin/activate
uv sync
```

Alternatively, you could install with python's built in venv and pip:
```bash
python3 -m venv .venv
source .venv/bin/activate 
# on Windows, instead do 
# .venv\Scripts\activate

pip install -e . # install in editable mode
```

### Environment Variables
You must set API keys to enable LLM analysis. Create a `.env` file in the project root with the following contents:
```bash
export OPENAI_API_KEY=your_openai_key_here  # Required for all classification
export COHERE_API_KEY=your_cohere_key_here  # Optional - only for legacy reranking (GPT reranker used by default)
```


## Additional Usage Patterns

View all options using `amc --help`. Here are some common usage patterns:

```bash
# Specify a different GPT model
amc --mission TESS --bibcode 2020MNRAS.491.2982E --gpt-model gpt-4.1-mini-2025-04-14

# Force reprocessing and save to different directory
amc --mission HST --bibcode 2020MNRAS.491.2982E --reprocess --output-dir ./results-reprocessed

# Use legacy Cohere reranker instead of GPT
amc --mission GALEX --bibcode 2020MNRAS.491.2982E --no-gpt-reranker

# Limit batch processing for testing
amc --mission TESS --batch-mode all --limit-papers 10

# Adjust classification thresholds
amc --mission JWST --bibcode 2020MNRAS.491.2982E --science-threshold 0.7 --reranker-threshold 0.1
```

## Key Options

**Required:**
-   `--mission MISSION`: Mission name (e.g., TESS, JWST, GALEX, PANSTARRS) for classification
-   `--bibcode BIBCODE` OR `--batch-mode BATCH_MODE`: Either a specific bibcode for single paper analysis, or batch mode ('all', file path, or comma-separated bibcodes)

**Data and Output:**
-   `--data-file DATA_FILE`: Path to JSON data file containing paper records (default: `data/combined_dataset_2025_03_25.json`)
-   `--output-dir OUTPUT_DIR, -o OUTPUT_DIR`: Directory where `results/` subdirectory will be created (default: current directory)
-   `--prompts-dir PROMPTS_DIR, -p PROMPTS_DIR`: Directory containing LLM prompt templates (default: `./prompts`)

**Analysis Parameters:**
-   `--science-threshold SCIENCE_THRESHOLD`: Threshold for classifying papers as mission science (0-1, default: `0.5`)
-   `--reranker-threshold RERANKER_THRESHOLD`: Minimum reranker score to proceed with LLM analysis (0-1, default: `0.05`)
-   `--top-k-snippets TOP_K_SNIPPETS`: Number of top reranked snippets to send to LLM (default: `5`)
-   `--context-sentences CONTEXT_SENTENCES`: Sentences before/after keyword sentences in snippets (default: `3`)

**Model Configuration:**
-   `--gpt-model GPT_MODEL`: OpenAI GPT model for classification (default: `gpt-4.1-mini-2025-04-14`)
-   `--cohere-reranker-model COHERE_RERANKER_MODEL`: Cohere reranker model (default: `rerank-v3.5`)
-   `--no-gpt-reranker`: Use legacy Cohere reranker instead of GPT-4.1-nano reranker

**Processing Options:**
-   `--reprocess`: Force reprocessing, ignoring caches
-   `--limit-papers LIMIT_PAPERS`: Limit processing to first N papers (batch mode only)

**API Keys:**
-   `--openai-key OPENAI_KEY`: OpenAI API key (or use `OPENAI_API_KEY` env var)
-   `--cohere-key COHERE_KEY`: Cohere API key (or use `COHERE_API_KEY` env var; only needed for legacy reranking)


## Outputs

### Batch Mode
Generates reports in the `results/` directory:
- `{data_filename}_report.json`: Summary report with paper counts and analysis results
- `{data_filename}_report.csv`: CSV export with all papers and their classification results
- Cache files: `{mission}_batch_science.json`, `{mission}_batch_snippets.json`, etc.

### Single Paper Mode
Outputs JSON to stdout with classification results, including:
- Mission relevance score and classification
- Supporting quotes from the paper
- Analysis metadata and timestamps

## Data Requirements

The tool expects a JSON data file containing paper records with the following structure:
```json
[
  {
    "bibcode": "2020MNRAS.491.2982E",
    "title": " HD 213885b: a transiting 1-d-period super-Earth with an Earth-like composition around a bright (V = 7.9) star unveiled by TESS",
    "body": "Full text content of the paper..."
  }
]
```

Other fields can also be included as metadata. The default data file is `data/combined_dataset_2025_03_25.json`.

