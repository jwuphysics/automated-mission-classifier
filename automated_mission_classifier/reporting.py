"""Report generation functionality."""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from .utils.cache import load_cache

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates analysis reports."""
    
    def __init__(self, 
                 results_dir: Path,
                 science_threshold: float,
                 model_config: Dict[str, str]):
        self.results_dir = results_dir
        self.science_threshold = science_threshold
        self.model_config = model_config
        
    def generate_report(self, batch_identifier: str, cache_files: Dict[str, Path], limit_papers: Optional[int] = None) -> Optional[Dict]:
        """Generate a summary report of the analysis."""
        science_results = load_cache(cache_files['science'])
        skipped_results = load_cache(cache_files['skipped'])
        papers_cache = load_cache(cache_files['papers'])

        total_attempted = len(papers_cache)
        successfully_processed = {
            k: v for k, v in science_results.items() 
            if isinstance(v, dict) and v.get("science", v.get("jwstscience", -1.0)) >= 0
        }
        analysis_failed_count = len(papers_cache) - len(successfully_processed) - len(skipped_results)

        science_papers_ids = {
            paper_id for paper_id, r in successfully_processed.items()
            if r.get("science", r.get("jwstscience", 0.0)) >= self.science_threshold
        }
        science_papers_count = len(science_papers_ids)

        detailed_results_list = []

        for paper_id in science_papers_ids:
            science_info = successfully_processed[paper_id]
            paper_info = papers_cache.get(paper_id, {})
            detailed_results_list.append({
                "bibcode": paper_info.get("bibcode", paper_id),
                "arxiv_id": paper_info.get("arxiv_id", ""),
                "arxiv_url": paper_info.get("arxiv_url", ""),
                "science_score": science_info.get("science", science_info.get("jwstscience")),
                "science_reason": science_info.get("reason"),
                "science_quotes": science_info.get("quotes")
            })

        metadata = {
            "report_generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "batch_analyzed": batch_identifier,
            "science_threshold": self.science_threshold,
            **self.model_config
        }
        
        if limit_papers is not None:
            metadata["limit_papers"] = limit_papers
        
        report = {
            "metadata": metadata,
            "summary": {
                "total_papers_in_dataset": total_attempted,
                "papers_skipped_before_analysis": len(skipped_results),
                "papers_analysis_failed": analysis_failed_count, 
                "papers_successfully_analyzed": len(successfully_processed), 
                "mission_science_papers_found": science_papers_count,
            },
            "skipped_papers_details": dict(sorted(skipped_results.items())), 
            "mission_science_papers_details": sorted(detailed_results_list, key=lambda x: x['bibcode'])
        }

        json_filename = f"{batch_identifier}_report"
        if limit_papers is not None:
            json_filename += f"_limit{limit_papers}"
        json_filename += ".json"
        report_path = self.results_dir / json_filename
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save report file {report_path}: {e}")

        logger.info(f"Report generated: {report_path}")
        
        # Print Summary
        logger.info(f"--- Summary for {batch_identifier} ---")
        logger.info(f"  Total papers in dataset: {report['summary']['total_papers_in_dataset']}")
        logger.info(f"  Skipped before analysis: {report['summary']['papers_skipped_before_analysis']}")
        logger.info(f"  Analysis Failed (LLM/etc): {report['summary']['papers_analysis_failed']}")
        logger.info(f"  Successfully Analyzed: {report['summary']['papers_successfully_analyzed']}")
        logger.info(f"  -> {self.model_config.get('mission', 'Mission')} Science Papers (Score >= {self.science_threshold}): {science_papers_count}")
        logger.info("--- End Summary ---")

        return report

    def generate_csv_report(self, batch_identifier: str, cache_files: Dict[str, Path], limit_papers: Optional[int] = None) -> Optional[Path]:
        """Generate a CSV report with ALL papers and their processing status."""
        science_results = load_cache(cache_files['science'])
        skipped_results = load_cache(cache_files['skipped'])
        downloaded_papers = load_cache(cache_files['downloaded'])
        papers_cache = load_cache(cache_files['papers'])
        
        # Create CSV data for ALL papers in the papers cache
        csv_data = []
        
        for paper_id, paper_info in papers_cache.items():
            # Start with basic paper info
            row = {
                "bibcode": paper_info.get("bibcode", paper_id),
                "arxiv_id": paper_info.get("arxiv_id", ""),
                "arxiv_url": paper_info.get("arxiv_url", ""),
                "paper_title": paper_info.get("title", ""),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "top_quotes": "",
                "science_score": 0.0,
                "reason": "",
                "status": ""
            }
            
            # Check if paper was skipped before analysis
            if paper_id in skipped_results:
                skip_info = skipped_results[paper_id]
                row.update({
                    "status": skip_info.get("reason", ""),
                    "timestamp": skip_info.get("timestamp", ""),
                    "reason": "Skipped before analysis"
                })
            
            # Check if paper had science analysis
            elif paper_id in science_results:
                science_info = science_results[paper_id]
                
                if isinstance(science_info, dict):
                    # Handle quotes formatting
                    quotes = science_info.get("quotes", [])
                    if isinstance(quotes, list):
                        quotes_str = "|".join(quotes)  # Join with pipe separator
                    else:
                        quotes_str = str(quotes)
                    
                    row.update({
                        "science_score": science_info.get("science", science_info.get("jwstscience", 0.0)),
                        "reason": science_info.get("reason", ""),
                        "top_quotes": quotes_str
                    })
                    
                    # Determine status based on science analysis
                    science_score = science_info.get("science", science_info.get("jwstscience", 0.0))
                    science_reason = science_info.get("reason", "")
                    
                    if science_score < 0:
                        row["status"] = "Science analysis failed"
                    elif "No relevant keywords" in science_reason:
                        row["status"] = f"No {self.model_config.get('mission', 'mission')} keywords found"
                    elif "reranker score" in science_reason:
                        row["status"] = "Below reranker threshold"
                    elif science_score < self.science_threshold:
                        row["status"] = "Below science threshold"
                    else:
                        row["status"] = f"{self.model_config.get('mission', 'Mission')} science paper"
                else:
                    row.update({
                        "status": "Invalid science analysis result",
                        "reason": "Analysis failed"
                    })
            
            # Paper wasn't processed at all
            else:
                row.update({
                    "status": "Not processed",
                    "reason": "Paper not processed"
                })
            
            csv_data.append(row)
        
        # Sort by bibcode
        csv_data.sort(key=lambda x: x['bibcode'])
        
        # Write CSV file
        csv_filename = f"{batch_identifier}_report"
        if limit_papers is not None:
            csv_filename += f"_limit{limit_papers}"
        csv_filename += ".csv"
        csv_path = self.results_dir / csv_filename
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ["bibcode", "arxiv_id", "arxiv_url", "paper_title", "top_quotes", 
                            "science_score", "reason", "timestamp", "status"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
                
            logger.info(f"CSV report generated: {csv_path}")
            return csv_path
            
        except Exception as e:
            logger.error(f"Failed to save CSV report file {csv_path}: {e}")
            return None