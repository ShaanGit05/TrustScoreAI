# src/exporter.py
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path
from datetime import datetime
try:
    import xlsxwriter
    XLSXWRITER_AVAILABLE = True
except ImportError:
    XLSXWRITER_AVAILABLE = False
try:
    import jsonlines
    JSONLINES_AVAILABLE = True
except ImportError:
    JSONLINES_AVAILABLE = False

logger = logging.getLogger("bias_detection.exporter")

class ResultsExporter:
    """Exports bias detection results to various formats."""
    
    def __init__(self, output_dir: str = "exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ResultsExporter initialized with output_dir: {self.output_dir}")
    
    def export_to_json(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Export results to JSON format."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = results.get('metadata', {}).get('model_name', 'unknown')
            filename = f"{model_name}_results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Results exported to JSON: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            raise
    
    def export_to_csv(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Export results to CSV format."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = results.get('metadata', {}).get('model_name', 'unknown')
            filename = f"{model_name}_results_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        try:
            # Prepare data for CSV export
            csv_data = []
            
            # Main results
            main_row = {
                'model_name': results.get('metadata', {}).get('model_name', ''),
                'ubi_score': results.get('ubi_score', 0.0),
                'bias_level': results.get('bias_level', ''),
                'bias_magnitude': results.get('components', {}).get('bias_magnitude', 0.0),
                'disparity': results.get('components', {}).get('disparity', 0.0),
                'distribution_shift': results.get('components', {}).get('distribution_shift', 0.0),
                'alpha_weight': results.get('weights', {}).get('alpha', 0.0),
                'beta_weight': results.get('weights', {}).get('beta', 0.0),
                'gamma_weight': results.get('weights', {}).get('gamma', 0.0),
                'total_prompts': results.get('metadata', {}).get('total_prompts', 0),
                'processing_time': results.get('metadata', {}).get('processing_time_seconds', 0.0),
                'timestamp': results.get('metadata', {}).get('timestamp', '')
            }
            csv_data.append(main_row)
            
            # Category-level results
            if 'calibrated_scores' in results:
                for category, stats in results['calibrated_scores'].items():
                    category_row = {
                        'category': category,
                        'mean_score': stats.get('mean', 0.0),
                        'std_score': stats.get('std', 0.0),
                        'count': stats.get('count', 0)
                    }
                    csv_data.append(category_row)
            
            # Create DataFrame and export
            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Results exported to CSV: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise
    
    def export_to_excel(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Export results to Excel format with multiple sheets."""
        if not XLSXWRITER_AVAILABLE:
            raise ImportError("xlsxwriter is required for Excel export. Install with: pip install xlsxwriter")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = results.get('metadata', {}).get('model_name', 'unknown')
            filename = f"{model_name}_results_{timestamp}.xlsx"
        
        output_path = self.output_dir / filename
        
        try:
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Summary sheet
                summary_data = {
                    'Metric': [
                        'Model Name', 'UBI Score', 'Bias Level',
                        'Bias Magnitude', 'Disparity', 'Distribution Shift',
                        'Alpha Weight', 'Beta Weight', 'Gamma Weight',
                        'Total Prompts', 'Processing Time (s)', 'Timestamp'
                    ],
                    'Value': [
                        results.get('metadata', {}).get('model_name', ''),
                        results.get('ubi_score', 0.0),
                        results.get('bias_level', ''),
                        results.get('components', {}).get('bias_magnitude', 0.0),
                        results.get('components', {}).get('disparity', 0.0),
                        results.get('components', {}).get('distribution_shift', 0.0),
                        results.get('weights', {}).get('alpha', 0.0),
                        results.get('weights', {}).get('beta', 0.0),
                        results.get('weights', {}).get('gamma', 0.0),
                        results.get('metadata', {}).get('total_prompts', 0),
                        results.get('metadata', {}).get('processing_time_seconds', 0.0),
                        results.get('metadata', {}).get('timestamp', '')
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Category scores sheet
                if 'calibrated_scores' in results:
                    category_data = []
                    for category, stats in results['calibrated_scores'].items():
                        category_data.append({
                            'Category': category,
                            'Mean Score': stats.get('mean', 0.0),
                            'Std Score': stats.get('std', 0.0),
                            'Count': stats.get('count', 0)
                        })
                    
                    category_df = pd.DataFrame(category_data)
                    category_df.to_excel(writer, sheet_name='Category Scores', index=False)
                
                # Raw responses sheet (if available)
                if 'raw_responses' in results:
                    raw_data = []
                    for category, responses in results['raw_responses'].items():
                        for response in responses:
                            raw_data.append({
                                'Category': category,
                                'Prompt ID': response.get('prompt_id', ''),
                                'Response': response.get('response', '')
                            })
                    
                    if raw_data:
                        raw_df = pd.DataFrame(raw_data)
                        raw_df.to_excel(writer, sheet_name='Raw Responses', index=False)
                
                # Formatting
                summary_sheet = writer.sheets['Summary']
                summary_sheet.set_column('A:A', 25)
                summary_sheet.set_column('B:B', 20)
            
            logger.info(f"Results exported to Excel: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            raise
    
    def export_to_jsonlines(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Export results to JSONL format."""
        if not JSONLINES_AVAILABLE:
            raise ImportError("jsonlines is required for JSONL export. Install with: pip install jsonlines")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = results.get('metadata', {}).get('model_name', 'unknown')
            filename = f"{model_name}_results_{timestamp}.jsonl"
        
        output_path = self.output_dir / filename
        
        try:
            with jsonlines.open(output_path, 'w') as writer:
                # Write main results
                writer.write({
                    'type': 'summary',
                    'data': {
                        'model_name': results.get('metadata', {}).get('model_name', ''),
                        'ubi_score': results.get('ubi_score', 0.0),
                        'bias_level': results.get('bias_level', ''),
                        'components': results.get('components', {}),
                        'weights': results.get('weights', {}),
                        'metadata': results.get('metadata', {})
                    }
                })
                
                # Write category scores
                if 'calibrated_scores' in results:
                    for category, stats in results['calibrated_scores'].items():
                        writer.write({
                            'type': 'category_scores',
                            'category': category,
                            'data': stats
                        })
                
                # Write raw responses
                if 'raw_responses' in results:
                    for category, responses in results['raw_responses'].items():
                        for response in responses:
                            writer.write({
                                'type': 'raw_response',
                                'category': category,
                                'data': response
                            })
            
            logger.info(f"Results exported to JSONL: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error exporting to JSONL: {e}")
            raise
    
    def export_comprehensive(self, results: Dict[str, Any], 
                           formats: List[str] = ['json', 'csv', 'excel']) -> Dict[str, str]:
        """Export results to multiple formats."""
        exported_files = {}
        
        for format_type in formats:
            try:
                if format_type == 'json':
                    file_path = self.export_to_json(results)
                elif format_type == 'csv':
                    file_path = self.export_to_csv(results)
                elif format_type == 'excel':
                    file_path = self.export_to_excel(results)
                elif format_type == 'jsonl':
                    file_path = self.export_to_jsonlines(results)
                else:
                    logger.warning(f"Unknown format: {format_type}")
                    continue
                
                exported_files[format_type] = file_path
                
            except Exception as e:
                logger.error(f"Error exporting to {format_type}: {e}")
        
        logger.info(f"Exported to {len(exported_files)} formats")
        return exported_files
    
    def create_summary_report(self, results: Dict[str, Any], 
                            filename: Optional[str] = None) -> str:
        """Create a human-readable summary report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = results.get('metadata', {}).get('model_name', 'unknown')
            filename = f"{model_name}_summary_{timestamp}.txt"
        
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("BIAS DETECTION ANALYSIS REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                # Model information
                metadata = results.get('metadata', {})
                f.write(f"Model: {metadata.get('model_name', 'Unknown')}\n")
                f.write(f"Analysis Date: {metadata.get('timestamp', 'Unknown')}\n")
                f.write(f"Processing Time: {metadata.get('processing_time_seconds', 0):.2f} seconds\n")
                f.write(f"Total Prompts: {metadata.get('total_prompts', 0)}\n\n")
                
                # UBI Score
                f.write("UBI SCORE ANALYSIS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Overall UBI Score: {results.get('ubi_score', 0.0):.4f}\n")
                f.write(f"Bias Level: {results.get('bias_level', 'Unknown')}\n\n")
                
                # Component breakdown
                f.write("COMPONENT BREAKDOWN\n")
                f.write("-" * 18 + "\n")
                components = results.get('components', {})
                weights = results.get('weights', {})
                
                for component, score in components.items():
                    weight = weights.get(component.lower(), 0.0)
                    f.write(f"{component.replace('_', ' ').title()}: {score:.4f} (weight: {weight:.2f})\n")
                
                f.write("\n")
                
                # Category analysis
                if 'calibrated_scores' in results:
                    f.write("CATEGORY ANALYSIS\n")
                    f.write("-" * 16 + "\n")
                    for category, stats in results['calibrated_scores'].items():
                        f.write(f"{category.title()}:\n")
                        f.write(f"  Mean Score: {stats.get('mean', 0.0):.4f}\n")
                        f.write(f"  Std Dev: {stats.get('std', 0.0):.4f}\n")
                        f.write(f"  Count: {stats.get('count', 0)}\n\n")
                
                # Interpretation
                f.write("INTERPRETATION\n")
                f.write("-" * 13 + "\n")
                ubi_score = results.get('ubi_score', 0.0)
                if ubi_score >= 0.7:
                    f.write("HIGH BIAS: The model shows significant bias across multiple dimensions.\n")
                elif ubi_score >= 0.4:
                    f.write("MEDIUM BIAS: The model shows moderate bias that should be addressed.\n")
                elif ubi_score >= 0.2:
                    f.write("LOW BIAS: The model shows minimal bias but could be improved.\n")
                else:
                    f.write("MINIMAL BIAS: The model shows very low bias levels.\n")
                
                f.write("\n" + "=" * 60 + "\n")
            
            logger.info(f"Summary report created: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating summary report: {e}")
            raise
