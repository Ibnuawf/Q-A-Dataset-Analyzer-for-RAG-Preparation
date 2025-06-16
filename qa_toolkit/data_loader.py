# qa_toolkit/data_loader.py
import pandas as pd
import json
import os
import logging
from typing import Dict, List, Optional, Any, Tuple # <--- ADD Tuple HERE
# from .text_processor import TextProcessor # Not used directly in this file based on provided snippet

logger = logging.getLogger(__name__)
METRICS_DICT_LOADER: Dict[str, Any] = {}

class DataLoader:
    def __init__(self, config: Dict):
        self.config = config
        self.paths_cfg = config['file_paths']
        self.loading_cfg = config['data_loading']
        self.column_map_cfg = config['column_mapping']

    def _extract_qa_from_nested_json(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        # ... (implementation from previous correct version) ...
        all_qa_records = []
        propagate_parent_title = self.loading_cfg.get('propagate_parent_category_title', False)

        for category_id, category_data in data.items():
            if isinstance(category_data, dict) and 'questions' in category_data:
                parent_title = category_data.get('title', f"Category_{category_id}")
                parent_description = category_data.get('description', "")

                for qa_record_orig in category_data.get('questions', []):
                    if isinstance(qa_record_orig, dict):
                        qa_record = qa_record_orig.copy()
                        if propagate_parent_title:
                            qa_record['parent_category_id'] = category_id
                            qa_record['parent_category_title'] = parent_title
                            qa_record['parent_category_description'] = parent_description
                        all_qa_records.append(qa_record)
            else:
                logger.warning(f"Skipping category_id '{category_id}' due to unexpected structure or missing 'questions' key.")
        return all_qa_records


    def _load_data_from_file(self, file_path: str) -> Optional[pd.DataFrame]:
        # ... (implementation from previous correct version) ...
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            if not isinstance(raw_data, dict):
                logger.error("Root of JSON is not a dictionary as expected. Cannot process.")
                return None
            qa_records_list = self._extract_qa_from_nested_json(raw_data)
            if not qa_records_list:
                logger.warning("No Q&A records extracted from the JSON structure.")
                return pd.DataFrame()
            df = pd.json_normalize(qa_records_list)
            return df
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing JSON file {file_path}: {e}", exc_info=True)
            return None

    def load_and_validate_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]: # Line 60
        # ... (implementation from previous correct version) ...
        input_file = self.paths_cfg['input_dataset']
        if not input_file or not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            raise FileNotFoundError(f"Input file not found: {input_file}")

        logger.info(f"Starting data loading for: {input_file}")
        os.makedirs(self.paths_cfg['output_dir'], exist_ok=True)
        df = self._load_data_from_file(input_file)
        if df is None:
            raise ValueError("Failed to load data from file into DataFrame.")

        file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
        use_sampling = file_size_mb > self.loading_cfg['max_file_size_full_load_mb']
        target_sample_size = self.loading_cfg['target_sample_size']
        METRICS_DICT_LOADER['initial_records_extracted'] = len(df)

        if use_sampling and not df.empty and len(df) > target_sample_size:
            logger.warning(
                f"File size ({file_size_mb:.2f}MB) and/or extracted records ({len(df)}) trigger sampling. "
                f"Sampling down to ~{target_sample_size} records."
            )
            df = df.sample(n=min(target_sample_size, len(df)), random_state=42, replace=False)
            logger.info(f"Sampled {len(df)} records.")
            METRICS_DICT_LOADER['records_after_sampling'] = len(df)
        
        if df.empty:
            logger.warning("DataFrame is empty after loading (and potential sampling).")
        else:
            logger.info(f"Proceeding with {len(df)} records for mapping and validation.")
            
        df = self._validate_and_map_columns(df)
        return df, METRICS_DICT_LOADER

    def _validate_and_map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... (implementation from previous correct version) ...
        col_map_cfg = self.column_map_cfg # Corrected: was self.config['column_mapping']
        rename_map = {
            actual_name: std_name
            for std_name, actual_name in col_map_cfg.items()
            if actual_name in df.columns
        }
        df = df.rename(columns=rename_map)
        logger.info(f"Columns after attempting rename: {df.columns.tolist()}")

        essential_cols = ['question', 'answer']
        missing_essential = [col for col in essential_cols if col not in df.columns]
        if missing_essential:
            msg = (f"Missing essential standardized columns after mapping: {missing_essential}. "
                f"Available columns in DataFrame: {df.columns.tolist()}. "
                f"Ensure 'column_mapping' in your config correctly maps your JSON fields "
                f" (e.g., 'question': 'your_actual_question_field_name').")
            logger.error(msg)
            raise ValueError(msg)

        id_std_name = 'id' 
        if id_std_name not in df.columns:
            logger.warning(f"Standardized ID column '{id_std_name}' not found. Creating a default sequential ID.")
            df[id_std_name] = range(len(df))
        
        if id_std_name in df.columns:
            if df[id_std_name].isnull().any():
                logger.warning(f"ID column '{id_std_name}' contains null values. These may affect uniqueness checks.")
            id_duplicates = df[id_std_name].dropna().duplicated().sum()
            if id_duplicates > 0:
                logger.warning(f"ID column '{id_std_name}' contains {id_duplicates} duplicate non-null values.")
        
        METRICS_DICT_LOADER['final_columns'] = df.columns.tolist()
        METRICS_DICT_LOADER['final_shape'] = df.shape
        logger.info(f"Columns after mapping and ID handling: {df.columns.tolist()}")
        return df