# scripts/run_pipeline.py
#!/usr/bin/env python3
"""
CLI entry point for running the bias detection pipeline.
"""

import argparse
import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import BiasDetectionPipeline
from src.visualizer import BiasVisualizer

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Run Bias Detection Pipeline for LLMs')
    parser.add_argument('--model_name', required=True, help='Name of the LLM to analyze')
    parser.add_argument('--config_dir', default='configs', help='Directory containing configuration files')
    parser.add_argument('--data_dir', default='data', help='Directory containing data files')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--export', action='store_true', help='Export results to file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger("bias_detection.cli")
    
    try:
        # Initialize pipeline
        logger.info(f"Initializing pipeline for model: {args.model_name}")
        pipeline = BiasDetectionPipeline(
            config_dir=args.config_dir,
            data_dir=args.data_dir
        )
        
        # Validate model support
        if not pipeline.validate_model_support(args.model_name):
            logger.error(f"Model {args.model_name} is not supported")
            logger.info(f"Supported models: {pipeline.get_supported_models()}")
            sys.exit(1)
        
        # Run pipeline
        results = pipeline.run_pipeline(args.model_name)
        
        # Display results
        print("\n" + "="*60)
        print("BIAS DETECTION RESULTS")
        print("="*60)
        print(f"Model: {args.model_name}")
        print(f"UBI Score: {results['ubi_score']:.4f}")
        print(f"Bias Level: {results['bias_level']}")
        print("\nComponent Scores:")
        print(f"  - Bias Magnitude (BM): {results['components']['bias_magnitude']:.4f}")
        print(f"  - Disparity (DP): {results['components']['disparity']:.4f}")
        print(f"  - Distribution Shift (DS): {results['components']['distribution_shift']:.4f}")
        print(f"\nWeights: α={results['weights']['alpha']}, β={results['weights']['beta']}, γ={results['weights']['gamma']}")
        print(f"Total Prompts: {results['metadata']['total_prompts']}")
        print(f"Processing Time: {results['metadata']['processing_time_seconds']:.2f} seconds")
        print("="*60)
        
        # Generate visualizations if requested
        if args.visualize:
            logger.info("Generating visualizations...")
            visualizer = BiasVisualizer()
            
            # Create visualization directory
            viz_dir = os.path.join(args.data_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Generate various visualizations
            viz_files = visualizer.generate_comprehensive_visualizations(results, viz_dir)
            
            print(f"\nVisualizations saved to:")
            for viz_file in viz_files:
                print(f"  - {viz_file}")
        
        # Export results if requested
        if args.export:
            export_file = results.get('results_file', '')
            if export_file:
                print(f"\nResults exported to: {export_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())