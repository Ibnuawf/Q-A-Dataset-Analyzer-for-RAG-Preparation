# main_analyzer.py
import os
import argparse
import logging
import warnings
from bs4 import MarkupResemblesLocatorWarning

# It's good practice to put package imports after standard library if possible,
# but for __init__ based imports, this is fine.
from qa_toolkit.config_loader import load_config
from qa_toolkit.data_loader import DataLoader
from qa_toolkit.analysis_engine import QAAnalyzer
from qa_toolkit.reporting import ReportGenerator

# Suppress BeautifulSoup MarkupResemblesLocatorWarning (Optional)
warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)

# Global metrics dictionary to be passed around and populated
GLOBAL_METRICS_DICT = {} 

def setup_logging(log_file_path: str, log_level=logging.INFO):
    """Configures logging for the application."""
    root_logger = logging.getLogger()
    # Remove any existing handlers to avoid duplicate logs if re-run in same session
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File Handler (always UTF-8)
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Stream Handler (for console)
    stream_handler = logging.StreamHandler()
    # Use a simpler format for console for brevity, or keep it the same
    stream_handler.setFormatter(logging.Formatter('%(levelname)s: [%(name)s] %(message)s'))
    root_logger.addHandler(stream_handler)
    
    root_logger.setLevel(log_level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING) # Quieten matplotlib's verbose INFO logs

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Q&A Dataset Analyzer (Pro Version).")
    parser.add_argument("input_file", type=str, help="Path to the JSON Q&A dataset file.")
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to a custom YAML configuration file (optional).")
    args = parser.parse_args()

    # Determine path to default_config.yaml relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_file = os.path.join(script_dir, "config", "default_config.yaml")

    # Initial basic logging to capture config loading issues
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    config = load_config(default_config_file, args.config)
    
    # Override input_file in config with CLI argument if provided
    if args.input_file:
        config['file_paths']['input_dataset'] = args.input_file
    elif not config['file_paths'].get('input_dataset'):
        logging.error("Input dataset file path must be provided either via CLI or in the config file.")
        return

    # Setup more detailed logging now that output_dir is known
    output_dir = config['file_paths']['output_dir']
    log_file_name = config['file_paths']['log_file']
    os.makedirs(output_dir, exist_ok=True) # Ensure output dir exists for log file
    log_file_path = os.path.join(output_dir, log_file_name)
    setup_logging(log_file_path) # Re-initialize logging with file handler

    main_logger = logging.getLogger("MainAnalyzerApp") # Get a specific logger
    main_logger.info(f"Professional Q&A Analyzer starting for: {config['file_paths']['input_dataset']}")
    main_logger.info(f"Using output directory: {output_dir}")

    try:
        data_loader = DataLoader(config)
        df, loader_metrics = data_loader.load_and_validate_data()
        GLOBAL_METRICS_DICT['data_loading'] = loader_metrics # Store loader metrics

        analyzer = QAAnalyzer(df, config, GLOBAL_METRICS_DICT) # Pass the global metrics dict
        html_report_sections = analyzer.run_all_analyses() # This now populates GLOBAL_METRICS_DICT

        reporter = ReportGenerator(config, html_report_sections, GLOBAL_METRICS_DICT)
        reporter.generate_all_reports()

        main_logger.info("Analysis completed successfully!")

    except FileNotFoundError as e:
        main_logger.error(f"File not found error: {e}", exc_info=True)
    except ValueError as e:
        main_logger.error(f"Data validation or processing error: {e}", exc_info=True)
    except Exception as e:
        error_type_name = type(e).__name__
        main_logger.error(f"An unexpected critical error occurred ({error_type_name}): {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()