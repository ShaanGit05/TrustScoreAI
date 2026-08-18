from pathlib import Path
import json
import logging
import os
import re
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for safe use in file paths. Replaces / : \\ * ? \" < > | with _."""
    if not model_name:
        return "unknown"
    s = str(model_name).replace("/", "_").replace(":", "_")
    return re.sub(r'[\\*?"<>|]', "_", s)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and processing of bias detection datasets."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.results_dir = self.data_dir / "results"
        self.baselines_dir = self.data_dir / "baselines"
        
        # Create directories if they don't exist
        for dir_path in [self.processed_dir, self.results_dir, self.baselines_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataLoader initialized with data_dir: {self.data_dir}")
    
    def find_data_file(self, filename: str) -> Optional[Path]:
        """Dynamically find the data file by searching common locations."""
        search_locations = [
            self.raw_dir / filename,
            Path(__file__).parent / filename,
            Path(__file__).parent.parent / "data" / "raw" / filename,
            Path.cwd() / "data" / "raw" / filename,
            Path.cwd() / filename,
        ]
        
        for location in search_locations:
            if location.exists():
                logger.info(f"Found file at: {location}")
                return location
        
        logger.error(f"File not found: {filename}")
        return None
    
    def load_json_file(self, filename: str) -> List[Dict[str, Any]]:
        """Load data from a JSON file."""
        file_path = self.find_data_file(filename)
        
        if not file_path:
            logger.error(f"Could not find {filename}")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return [entry for entry in data if isinstance(entry, dict)]
            elif isinstance(data, dict):
                return [data]
            else:
                logger.warning(f"Unexpected data type in {filename}: {type(data)}")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {filename}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return []
    
    def load_all_bias_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load all bias detection datasets."""
        datasets = {}
        
        # Define dataset files
        dataset_files = {
            'race': 'race_prompt.json',
            'gender': 'gender_prompt.json', 
            'profession': 'profession_prompt.json',
            'religious_ideology': 'religious_ideology_prompt.json',
            'political_ideology': 'political_ideology_prompt.json'
        }
        
        for category, filename in dataset_files.items():
            data = self.load_json_file(filename)
            if data:
                datasets[category] = data
                logger.info(f"Loaded {len(data)} entries for {category}")
            else:
                logger.warning(f"No data loaded for {category}")
        
        return datasets
    
    def process_prompts(self, raw_prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process raw prompts into standardized format."""
        processed = []
        
        for i, prompt in enumerate(raw_prompts):
            processed_prompt = {
                'id': prompt.get('id', f"prompt_{i}"),
                'text': prompt.get('text', ''),
                'category': prompt.get('category', 'unknown'),
                'subcategory': prompt.get('subcategory', ''),
                'expected_bias': prompt.get('expected_bias', 0.0),
                'metadata': prompt.get('metadata', {})
            }
            processed.append(processed_prompt)
        
        return processed
    
    def save_processed_dataset(self, category: str, processed_data: List[Dict[str, Any]]) -> str:
        """Save processed dataset to file."""
        output_file = self.processed_dir / f"{category}_processed.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved processed {category} dataset to {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error saving processed {category} dataset: {e}")
            raise
    
    def save_results(self, results: Dict[str, Any], model_name: str) -> str:
        """Save analysis results to file. Uses sanitized model name for filename safety."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = sanitize_model_name(model_name)
        filename = f"{safe_name}_results_{timestamp}.json"
        output_file = self.results_dir / filename
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved results to {output_file}")
            return str(output_file)
        except Exception as e:
            msg = f"Failed to save results: {e}"
            logger.error(msg)
            raise OSError(msg)
    
    def load_processed_dataset(self, category: str) -> List[Dict[str, Any]]:
        """Load processed dataset."""
        processed_file = self.processed_dir / f"{category}_processed.json"
        
        if not processed_file.exists():
            logger.warning(f"Processed dataset not found: {processed_file}")
            return []
        
        try:
            with open(processed_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data if isinstance(data, list) else []
            
        except Exception as e:
            logger.error(f"Error loading processed {category} dataset: {e}")
            return []


# Simple utility API expected by tests (backward-compatible)
def load_json(path: str) -> List[Dict[str, Any]]:
    """Load a JSON file from an explicit path and return list of records."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return [entry for entry in data if isinstance(entry, dict)]
        if isinstance(data, dict):
            return [data]
        logger.warning(f"Unexpected data type in {path}: {type(data)}")
        return []
    except Exception as e:
        logger.error(f"Error loading json from {path}: {e}")
        return []


def validate_dataset(records: List[Dict[str, Any]]) -> None:
    """Basic validation: ensure list of dicts with required keys if present.
    Raises on grossly malformed input; otherwise no-op (tests expect no raise).
    """
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    for rec in records:
        if not isinstance(rec, dict):
            raise ValueError("each record must be a dict")
        # Accept either 'prompt' or 'text' as the prompt field
        if 'prompt' not in rec and 'text' not in rec:
            raise ValueError("record missing 'prompt' or 'text'")


def prepare_prompts(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform raw records into standardized prompt objects used downstream.
    Expected by tests to include 'id' and 'prompt_text'.
    """
    processed: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        prompt_text = rec.get('prompt') or rec.get('text') or ''
        processed.append({
            'id': rec.get('id', f"prompt_{i}"),
            'prompt_text': prompt_text,
            'category': rec.get('category', rec.get('protected_attribute', 'unknown')),
            'subcategory': rec.get('subcategory', rec.get('protected_value', '')),
            'expected_bias': rec.get('expected_bias', 0.0),
            'metadata': rec.get('metadata', {})
        })
    return processed


def save_processed(processed: List[Dict[str, Any]], path: str) -> None:
    """Write processed list to a JSON file at the provided path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)


# Legacy function for backward compatibility
def find_data_file(filename="race_prompt.json"):
    """Legacy function - use DataLoader.find_data_file instead."""
    loader = DataLoader()
    return loader.find_data_file(filename)

def load_race_prompts():
    """Legacy function - use DataLoader.load_json_file instead."""
    loader = DataLoader()
    return loader.load_json_file("race_prompt.json")


if __name__ == "__main__":
    print("Testing Data Loader...")
    
    loader = DataLoader()
    datasets = loader.load_all_bias_datasets()
    
    for category, data in datasets.items():
        print(f"\n{category}: {len(data)} entries")
        if data:
            print(f"Sample entry: {data[0]}")