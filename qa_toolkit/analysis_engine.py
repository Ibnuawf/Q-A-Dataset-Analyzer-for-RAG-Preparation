# qa_toolkit/analysis_engine.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
import textwrap
import os
import logging
from typing import Dict, List, Any, Tuple, Optional, Union, Set

from .text_processor import TextProcessor # Relative import
from .utils import save_plot_and_get_uri  # Relative import

logger = logging.getLogger(__name__)

class QAAnalyzer:
    def __init__(self, df: pd.DataFrame, config: Dict, global_metrics_dict: Dict):
        self.df = df.copy()
        self.config = config
        self.paths_cfg = config['file_paths']
        self.analysis_cfg = config['analysis_params']
        self.steps_cfg = config['analysis_steps_to_run']
        self.report_cfg = config['reporting']
        
        self.html_report_parts: List[Dict[str, Any]] = []
        self.global_metrics_dict = global_metrics_dict # Use the shared metrics dict

    def _add_to_report_and_metrics(self, 
                                   section_title: str, 
                                   metric_key_base: str,
                                   text_content: Union[str, List[str]],
                                   metrics_data: Optional[Dict[str, Any]] = None,
                                   html_table: Optional[str] = None, 
                                   plot_filename_stem: Optional[str] = None, 
                                   plot_figure: Optional[plt.Figure] = None,
                                   is_code_block: bool = False):
        """Helper to add to report structures and global metrics."""
        # HTML Report Part
        html_part = {"title": section_title, "type": "text", "text_content": text_content}
        plot_uri = None
        if plot_figure and plot_filename_stem:
            plot_uri = save_plot_and_get_uri(
                plot_figure, plot_filename_stem, 
                self.paths_cfg['output_dir'], 
                self.paths_cfg['plot_img_dir'],
                self.report_cfg['embed_plots_in_html'],
                self.report_cfg['save_individual_plots']
            )
        
        if html_table:
            html_part["type"] = "html_table"; html_part["html_content"] = html_table
        elif plot_uri:
            html_part["type"] = "plot"; html_part["plot_uri"] = plot_uri
            if metrics_data is not None: metrics_data['plot_path'] = plot_uri # Store path/URI
        elif is_code_block:
             html_part["type"] = "code_block"
             if isinstance(text_content, list): html_part["text_content"] = "\n".join(map(str, text_content))
        
        self.html_report_parts.append(html_part)

        # Global Metrics
        if metrics_data is not None:
            self.global_metrics_dict.setdefault(metric_key_base, {}).update(metrics_data)


    def run_all_analyses(self) -> List[Dict[str, Any]]: # Return structured HTML parts
        logger.info("Starting all configured analyses...")
        if self.df.empty:
            logger.warning("DataFrame is empty. Most analyses will be skipped.")
            self._add_to_report_and_metrics("Analysis Status", "status", 
                                            "DataFrame is empty. Analysis cannot proceed fully.")
            return self.html_report_parts

        self._preprocess_data() # This method also updates METRICS_DICT

        if self.steps_cfg.get('basic_overview', False): self._analyze_basic_overview()
        if self.steps_cfg.get('content_quality', False): self._analyze_content_quality()
        if self.steps_cfg.get('duplicate_analysis', False): self._analyze_duplicates()
        if self.steps_cfg.get('metadata_and_signals', False): self._analyze_metadata_and_signals()
        if self.steps_cfg.get('question_type_analysis', False): self._analyze_question_types()
        if self.steps_cfg.get('rag_recommendations', False): self._generate_rag_specific_recommendations()
        
        logger.info("All configured analyses complete.")
        return self.html_report_parts

    def _preprocess_data(self):
        logger.info("Preprocessing data: HTML cleaning, length calculation, metadata typing...")
        # Ensure 'question' and 'answer' are strings and fill NaNs
        for col_std_name in ['question', 'answer']: # Standardized names
            if col_std_name in self.df.columns:
                self.df[col_std_name] = self.df[col_std_name].astype(str).fillna('')
            else: 
                logger.error(f"Essential standardized column '{col_std_name}' missing at preprocessing stage!")
                self.df[col_std_name] = pd.Series([""] * len(self.df), index=self.df.index) # Add empty series

        # Use standardized column names 'question' and 'answer'
        if self.analysis_cfg.get('perform_html_cleaning', True):
            self.df['question_clean'] = self.df['question'].apply(TextProcessor.clean_html_content)
            self.df['answer_clean'] = self.df['answer'].apply(TextProcessor.clean_html_content)
        else:
            self.df['question_clean'] = self.df['question']
            self.df['answer_clean'] = self.df['answer']

        self.df['question_length_words'] = self.df['question_clean'].apply(lambda x: len(x.split()) if pd.notna(x) else 0)
        self.df['answer_length_words'] = self.df['answer_clean'].apply(lambda x: len(x.split()) if pd.notna(x) else 0)

        # Standardize metadata types (using standardized column names)
        if 'timestamp' in self.df.columns: self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        if 'upvotes' in self.df.columns: self.df['upvotes'] = pd.to_numeric(self.df['upvotes'], errors='coerce')
        
        # Handle 'tags' based on standardized name 'tags' (if mapped)
        # The actual column in df might be 'tags' if mapping happened, or original if not.
        # Standard practice is to refer to standardized names after mapping.
        std_tags_col = 'tags' # Standardized name for tags column
        if std_tags_col in self.df.columns: 
            self.df['tags_normalized'] = self.df[std_tags_col].apply(
                lambda t_val: [tag.strip().lower() for tag in (t_val if isinstance(t_val, list) else str(t_val).split(',')) if isinstance(tag, str) and tag.strip()] if pd.notna(t_val) else []
            )
        
        self.global_metrics_dict.setdefault('preprocessing', {})['html_cleaned'] = self.analysis_cfg.get('perform_html_cleaning', True)
        logger.info("Preprocessing complete.")

    def _analyze_basic_overview(self):
        logger.info("Analyzing basic dataset overview...")
        total_records = len(self.df)
        
        metrics_data = self.global_metrics_dict.get('dataset_info', {}) # Get loader metrics
        metrics_data.update({ # Add/update with analyzer view
            'total_records_analyzed_by_engine': total_records,
        })
        if 'id' in self.df.columns: # Standardized ID
            metrics_data['id_column_stats'] = {
                'unique_ids': int(self.df['id'].nunique(dropna=False)),
                'duplicate_id_count': int(self.df['id'].dropna().duplicated().sum())
            }
        
        overview_text = [
            f"Total records analyzed by engine: {total_records}",
            f"Input file: {os.path.basename(self.paths_cfg['input_dataset'])}",
            f"Columns in DataFrame: {', '.join(self.df.columns.tolist())}",
        ]
        if 'id_column_stats' in metrics_data:
             overview_text.append(f"Unique IDs: {metrics_data['id_column_stats']['unique_ids']} (Duplicate non-null IDs: {metrics_data['id_column_stats']['duplicate_id_count']})")

        sample_html_table = None
        if self.report_cfg['generate_html_report'] and not self.df.empty:
            display_cols_std = ['id', 'question_clean', 'answer_clean', 'category', 'qna_title', 'summary', 'upvotes', 'source']
            sample_df_display = self.df[[col for col in display_cols_std if col in self.df.columns]].sample(min(5, len(self.df)), random_state=42)
            for col in ['question_clean', 'answer_clean', 'summary']: 
                if col in sample_df_display.columns:
                    sample_df_display[col] = sample_df_display[col].apply(lambda x: textwrap.shorten(str(x), width=100, placeholder="..."))
            sample_html_table = sample_df_display.to_html(classes=['table', 'table-striped', 'table-sm'], index=False, border=0, na_rep='-')
        
        self._add_to_report_and_metrics("Basic Dataset Overview", "dataset_overview", 
                                        overview_text, metrics_data=metrics_data, html_table=sample_html_table)

    def _analyze_content_quality(self):
        logger.info("Analyzing content quality...")
        if self.df.empty:
            self._add_to_report_and_metrics("Content Quality Analysis", "content_quality", "DataFrame empty.")
            return

        metrics_cq = {}
        # Missing content
        missing_q = self.df['question_clean'].apply(lambda x: not str(x).strip()).sum()
        missing_a = self.df['answer_clean'].apply(lambda x: not str(x).strip()).sum()
        metrics_cq.update({'empty_questions_count': int(missing_q),
                           'empty_questions_percent': round((missing_q / len(self.df)) * 100, 1),
                           'empty_answers_count': int(missing_a),
                           'empty_answers_percent': round((missing_a / len(self.df)) * 100, 1)})
        text_content = [f"Empty Qs: {metrics_cq['empty_questions_count']} ({metrics_cq['empty_questions_percent']}%)",
                        f"Empty As: {metrics_cq['empty_answers_count']} ({metrics_cq['empty_answers_percent']}%)"]

        # Short content
        short_q_df = self.df[self.df['question_length_words'] < self.analysis_cfg['short_question_words']]
        short_a_df = self.df[self.df['answer_length_words'] < self.analysis_cfg['short_answer_words']]
        metrics_cq.update({'short_questions_count': len(short_q_df), 
                           'short_questions_percent': round((len(short_q_df) / len(self.df)) * 100, 1),
                           'short_answers_count': len(short_a_df),
                           'short_answers_percent': round((len(short_a_df) / len(self.df)) * 100, 1)})
        text_content.extend([f"\nShort Qs (<{self.analysis_cfg['short_question_words']} words): {metrics_cq['short_questions_count']} ({metrics_cq['short_questions_percent']}%)",
                             f"Short As (<{self.analysis_cfg['short_answer_words']} words): {metrics_cq['short_answers_count']} ({metrics_cq['short_answers_percent']}%)"])
        
        # Length Distributions
        q_len_series = self.df['question_length_words']
        a_len_series = self.df['answer_length_words']
        q_len_desc = q_len_series.describe() if not q_len_series.empty else pd.Series(dtype='float64').describe()
        a_len_desc = a_len_series.describe() if not a_len_series.empty else pd.Series(dtype='float64').describe()
        metrics_cq.update({'question_length_stats': q_len_desc.round(2).to_dict(),
                           'answer_length_stats': a_len_desc.round(2).to_dict()})
        text_content.extend(["\nLENGTH DISTRIBUTIONS (WORDS):", "  Questions:\n" + q_len_desc.to_string(), "\n  Answers:\n" + a_len_desc.to_string()])
        
        self._add_to_report_and_metrics("Content Quality (Lengths, Missing, Short)", "content_quality",
                                        text_content, metrics_data=metrics_cq, is_code_block=True)
        
        # Plots for length distributions
        plot_fig_len = None
        if not self.df.empty and 'question_length_words' in self.df.columns and 'answer_length_words' in self.df.columns:
            fig_len, axs_len = plt.subplots(1, 2, figsize=(16, 6))
            if not q_len_series.empty:
                sns.histplot(q_len_series, bins=50, kde=True, ax=axs_len[0], edgecolor='black')
                axs_len[0].set_title('Question Lengths (Words)')
                if 'median' in q_len_desc and pd.notna(q_len_desc['median']):
                    axs_len[0].axvline(q_len_desc['median'], color='r', linestyle='--', label=f"Median: {q_len_desc['median']:.0f}")
                axs_len[0].legend()
            else: axs_len[0].set_title('Question Lengths - No Data')

            if not a_len_series.empty:
                sns.histplot(a_len_series, bins=50, kde=True, ax=axs_len[1], edgecolor='black')
                axs_len[1].set_title('Answer Lengths (Words)')
                if 'median' in a_len_desc and pd.notna(a_len_desc['median']):
                    axs_len[1].axvline(a_len_desc['median'], color='r', linestyle='--', label=f"Median: {a_len_desc['median']:.0f}")
                if 'long_answer_words_chunk_threshold' in self.analysis_cfg:
                     axs_len[1].axvline(self.analysis_cfg['long_answer_words_chunk_threshold'], color='g', linestyle=':', label=f"Chunk Threshold ({self.analysis_cfg['long_answer_words_chunk_threshold']})")
                axs_len[1].legend()
            else: axs_len[1].set_title('Answer Lengths - No Data')

            fig_len.suptitle("Content Length Distributions", fontsize=16)
            fig_len.tight_layout(rect=[0, 0, 1, 0.95])
            plot_fig_len = fig_len # Assign figure for reporting
        
        self._add_to_report_and_metrics("Length Distribution Plots", "content_quality_plots", # Use a different metric key for plots
                                        "Visual representation of question and answer lengths.", 
                                        plot_figure=plot_fig_len, plot_filename_stem="length_distributions")


    def _analyze_duplicates(self):
        # (Largely same as V3.2, ensure metric_key_base and self.global_metrics_dict are used)
        logger.info("Analyzing duplicate content...")
        if self.df.empty or len(self.df) < 2:
            self._add_to_report_and_metrics("Duplicate Analysis", "duplicate_analysis", "DataFrame too small/empty.")
            return

        metrics_dup = {}
        q_exact_dup_series = self.df.duplicated(subset=['question_clean'], keep=False)
        a_exact_dup_series = self.df.duplicated(subset=['answer_clean'], keep=False)
        metrics_dup.update({'exact_duplicate_questions_records_count': int(q_exact_dup_series.sum()),
                            'exact_duplicate_questions_records_percent': round(q_exact_dup_series.mean() * 100, 1),
                            'exact_duplicate_answers_records_count': int(a_exact_dup_series.sum()),
                            'exact_duplicate_answers_records_percent': round(a_exact_dup_series.mean() * 100, 1)})
        text_content = [f"Records with exact duplicate Q text: {metrics_dup['exact_duplicate_questions_records_count']} ({metrics_dup['exact_duplicate_questions_records_percent']}%)",
                        f"Records with exact duplicate A text: {metrics_dup['exact_duplicate_answers_records_count']} ({metrics_dup['exact_duplicate_answers_records_percent']}%)"]

        sample_size = min(self.analysis_cfg['near_duplicate_sample_size'], len(self.df))
        if sample_size < 2:
            text_content.append("\nSample size too small for near-duplicate analysis.")
            metrics_dup['near_duplicate_analysis_skipped'] = True
        else:
            sample_df = self.df.sample(n=sample_size, random_state=42)
            for content_type in ['question', 'answer']:
                col_name = f'{content_type}_clean'
                pairs, groups = self._find_near_duplicate_pairs_and_groups(sample_df, col_name)
                metrics_dup[f'near_duplicate_{content_type}s_sample_size'] = sample_size
                metrics_dup[f'near_duplicate_{content_type}_pairs_in_sample_count'] = len(pairs)
                metrics_dup[f'near_duplicate_{content_type}_groups_in_sample_count'] = len(groups)
                text_content.append(f"\nNear-duplicate {content_type}s (Jaccard > {self.analysis_cfg['near_duplicate_jaccard_threshold']:.2f}) in sample of {sample_size}:")
                text_content.extend([f"  - Pairs found: {len(pairs)}", f"  - Distinct groups: {len(groups)}"])
                if groups: text_content.append(f"  - Largest group size: {max(len(g) for g in groups)}")
        
        self._add_to_report_and_metrics("Duplicate Content Analysis", "duplicate_analysis",
                                        text_content, metrics_data=metrics_dup, is_code_block=True)

    def _find_near_duplicate_pairs_and_groups(self, sample_df: pd.DataFrame, column_name: str) -> Tuple[List[Tuple[Any, Any, float]], List[Set[Any]]]:
        # (Identical to V3.2 implementation)
        texts = sample_df[column_name].tolist(); original_indices = sample_df.index.tolist() 
        adj = defaultdict(list); pairs_found: List[Tuple[Any, Any, float]] = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if not texts[i] or not texts[j]: continue 
                sim = TextProcessor.calculate_jaccard_similarity(texts[i], texts[j]) # Use class TextProcessor
                if sim >= self.analysis_cfg['near_duplicate_jaccard_threshold']:
                    idx1, idx2 = original_indices[i], original_indices[j]
                    pairs_found.append((idx1, idx2, sim)); adj[idx1].append(idx2); adj[idx2].append(idx1)
        visited: Set[Any] = set(); groups: List[Set[Any]] = []
        for i in range(len(original_indices)):
            curr_idx = original_indices[i]
            if curr_idx not in visited and curr_idx in adj:
                group: Set[Any] = set(); queue: List[Any] = [curr_idx]; visited.add(curr_idx)
                head = 0
                while head < len(queue):
                    u = queue[head]; head += 1; group.add(u)
                    for v in adj.get(u, []):
                        if v not in visited: visited.add(v); queue.append(v)
                if len(group) > 1: groups.append(group)
            elif curr_idx not in visited: visited.add(curr_idx)
        return pairs_found, groups

    def _analyze_metadata_and_signals(self):
        # (Largely same as V3.2, ensure metric_key_base and self.global_metrics_dict are used, and plots assigned to fig vars)
        logger.info("Analyzing metadata and signals...")
        if self.df.empty: 
            self._add_to_report_and_metrics("Metadata & Signals Analysis", "metadata_signals", "DataFrame empty.")
            return

        text_content_meta = []
        metrics_meta = {}
        
        # Standardized column names from config, mapped to actual DataFrame column names
        cols_to_analyze_std_names = {'category': 'category', 'tags': 'tags_normalized', 
                                     'timestamp': 'timestamp', 'upvotes': 'upvotes'}

        for std_name, actual_col_name_in_df in cols_to_analyze_std_names.items():
            plot_fig = None # Initialize plot figure for this metadata item
            
            if actual_col_name_in_df not in self.df.columns or self.df[actual_col_name_in_df].isnull().all():
                text_content_meta.append(f"\n{std_name.capitalize()} data (expected in '{actual_col_name_in_df}') not available or all missing.")
                metrics_meta[f'{std_name}_stats'] = {'status': 'Not available or all missing'}
                continue
            
            text_content_meta.append(f"\n{std_name.capitalize().replace('_', ' ')} ANALYSIS:")
            series = self.df[actual_col_name_in_df].dropna() # Drop NaNs for most analyses, category handles it
            
            current_metrics = {}
            if series.empty and std_name != 'tags': # Tags can be empty lists
                text_content_meta.append("  No valid data after dropping NaNs.")
                current_metrics['status'] = 'No valid data after NaNs dropped'
            else:
                if std_name == 'category':
                    counts = self.df[actual_col_name_in_df].value_counts(dropna=False)
                    top_n = counts.head(self.analysis_cfg['max_categories_plot'])
                    text_content_meta.append(top_n.to_string())
                    current_metrics.update({'counts_top': top_n.to_dict(), 'unique': int(counts.nunique()), 'missing': int(self.df[actual_col_name_in_df].isnull().sum())})
                    if not top_n.empty:
                        fig, ax = plt.subplots(figsize=(10, max(5, len(top_n) * 0.4)))
                        top_n.plot(kind='barh', ax=ax, color=sns.color_palette("viridis", len(top_n))).invert_yaxis()
                        ax.set_title(f'Top {len(top_n)} Categories'); fig.tight_layout(); plot_fig = fig
                
                elif std_name == 'tags': # Uses 'tags_normalized'
                    all_items = [item for sublist in self.df[actual_col_name_in_df] if isinstance(sublist, list) for item in sublist]
                    counts = Counter(all_items)
                    top_n_list = counts.most_common(self.analysis_cfg['max_tags_plot'])
                    for item, count_val in top_n_list: text_content_meta.append(f"  {item}: {count_val}")
                    current_metrics.update({'counts_top': dict(top_n_list), 'unique': len(counts), 'records_no_tags': int((self.df[actual_col_name_in_df].apply(len) == 0).sum())})
                    if top_n_list:
                        fig, ax = plt.subplots(figsize=(10, max(5, len(top_n_list) * 0.4)))
                        sns.barplot(x=[c[1] for c in top_n_list], y=[c[0] for c in top_n_list], ax=ax, palette="mako")
                        ax.set_title(f'Top {len(top_n_list)} Tags'); fig.tight_layout(); plot_fig = fig

                elif std_name == 'timestamp':
                    text_content_meta.append(f"  Date Range: {series.min()} to {series.max()}")
                    monthly_activity = series.dt.to_period('M').value_counts().sort_index()
                    current_metrics.update({'min_date': str(series.min()), 'max_date': str(series.max()), 'records_with_ts': len(series), 'monthly_avg': round(monthly_activity.mean(),1) if not monthly_activity.empty else 0})
                    if not monthly_activity.empty:
                        fig, ax = plt.subplots(figsize=(12, 6)); monthly_activity.plot(kind='line', ax=ax, marker='o')
                        ax.set_title('Activity Over Time (Monthly)'); fig.tight_layout(); plot_fig = fig
                
                elif std_name == 'upvotes':
                    desc = series.describe()
                    text_content_meta.append(desc.to_string())
                    current_metrics.update(desc.round(2).to_dict())
                    current_metrics['missing_count'] = int(self.df[actual_col_name_in_df].isnull().sum())
                    if not series.empty:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.histplot(series, bins=30, kde=True, ax=ax, edgecolor='black')
                        ax.set_title('Upvotes Distribution')
                        if 'median' in desc and pd.notna(desc['median']): ax.axvline(desc['median'], color='r', linestyle='--', label=f"Median: {desc['median']:.0f}")
                        ax.legend(); fig.tight_layout(); plot_fig = fig
            
            metrics_meta[f'{std_name}_stats'] = current_metrics
            if plot_fig: # If a plot was generated for this metadata item
                 self._add_to_report_and_metrics(f"{std_name.capitalize()} Distribution Plot", f"metadata_plots_{std_name}", 
                                                f"Visual representation of {std_name}.", 
                                                plot_figure=plot_fig, plot_filename_stem=f"{std_name}_distribution")
        
        self._add_to_report_and_metrics("Metadata & Signals Analysis (Text Summary)", "metadata_signals",
                                        text_content_meta, metrics_data=metrics_meta, is_code_block=True)


    def _analyze_question_types(self):
        # (Largely same as V3.2, ensure metric_key_base and self.global_metrics_dict are used, assign plot to fig var)
        logger.info("Analyzing question types...")
        if self.df.empty or 'question_clean' not in self.df.columns:
            self._add_to_report_and_metrics("Basic Question Type Analysis", "question_types", "Skipped.")
            return

        self.df['question_type_basic'] = self.df['question_clean'].apply(TextProcessor.classify_question_type_basic)
        q_type_counts = self.df['question_type_basic'].value_counts()
        q_type_percent = self.df['question_type_basic'].value_counts(normalize=True).mul(100).round(1)
        df_q_type_dist = pd.DataFrame({'Count': q_type_counts, 'Percentage (%)': q_type_percent})
        text_content = ["Basic Question Type Distribution:", df_q_type_dist.to_string()]
        metrics_data = {'distribution': df_q_type_dist.to_dict('index')}
        
        plot_fig_qtype = None
        if not q_type_counts.empty:
            fig, ax = plt.subplots(figsize=(10, max(5, len(q_type_counts) * 0.4)))
            q_type_counts.plot(kind='barh', ax=ax, color=sns.color_palette("cubehelix", len(q_type_counts))).invert_yaxis()
            ax.set_title('Basic Question Type Distribution'); ax.set_xlabel('Count'); fig.tight_layout()
            plot_fig_qtype = fig

        self._add_to_report_and_metrics("Basic Question Type Analysis", "question_types",
                                        text_content, metrics_data=metrics_data, 
                                        plot_figure=plot_fig_qtype, plot_filename_stem="question_type_distribution",
                                        is_code_block=True)

    def _generate_rag_specific_recommendations(self):
        # (Largely same as V3.2, ensure metric_key_base and self.global_metrics_dict are used)
        logger.info("Generating RAG recommendations...")
        if self.df.empty: 
            self._add_to_report_and_metrics("RAG Recommendations", "rag_recommendations", "DataFrame empty.")
            return
            
        metrics_rag = {}
        rag_text = ["Insights & recommendations for RAG pipeline:"]
        cq_m = self.global_metrics_dict.get('content_quality', {}); dup_m = self.global_metrics_dict.get('duplicate_analysis', {})
        meta_m = self.global_metrics_dict.get('metadata_signals', {}); qtype_m_dist = self.global_metrics_dict.get('question_types', {}).get('distribution',{})

        usable_pairs_df = self.df[(self.df['question_clean'].str.strip() != '') & (self.df['answer_clean'].str.strip() != '')]
        usable_pairs_count = len(usable_pairs_df)
        metrics_rag['usable_pairs_count'] = usable_pairs_count
        rag_text.extend([f"\n**Data Quantity & Quality:**",
                         f"  - Total records: {len(self.df)}. Usable pairs (non-empty Q&A): {usable_pairs_count}.",
                         f"  - Short Qs: {cq_m.get('short_questions_count',0)} ({cq_m.get('short_questions_percent',0.0)}%)",
                         f"  - Short As: {cq_m.get('short_answers_count',0)} ({cq_m.get('short_answers_percent',0.0)}%)"])
        
        long_answers_count = self.df[self.df['answer_length_words'] > self.analysis_cfg['long_answer_words_chunk_threshold']].shape[0] if 'answer_length_words' in self.df.columns else 0
        metrics_rag['answers_needing_chunking_count'] = long_answers_count
        rag_text.append(f"  - Answers needing chunking (>{self.analysis_cfg['long_answer_words_chunk_threshold']} words): {long_answers_count}.")

        rag_text.extend([f"\n**Duplicates & Redundancy:**",
                         f"  - Exact duplicate Qs affect {dup_m.get('exact_duplicate_questions_records_percent',0.0)}% of records."])
        if 'near_duplicate_question_groups_in_sample_count' in dup_m: # Check if key exists
            rag_text.append(f"  - Near-duplicate Q analysis (sample) found {dup_m.get('near_duplicate_question_groups_in_sample_count','N/A')} groups.")

        rag_text.append(f"\n**Content Nature (Question Types):**")
        if qtype_m_dist:
            common_q_types = sorted(qtype_m_dist.items(), key=lambda item: item[1]['Percentage (%)'], reverse=True)[:3]
            rag_text.append(f"  - Dominant Q types: {', '.join([f'{qt[0]} ({qt[1]['Percentage (%)']:.1f}%)' for qt in common_q_types])}.")

        rag_text.append("\n**Actionable Recommendations:**")
        recs = ["1. Filter empty/short Q&A.", "2. Deduplicate Qs (exact & near).",
                f"3. Chunk {long_answers_count} long answers (>{self.analysis_cfg['long_answer_words_chunk_threshold']} words).", "4. Metadata Integration:"]
        cat_s = meta_m.get('category_stats', {}); tag_s = meta_m.get('tag_stats', {})
        up_s = meta_m.get('upvotes_stats', {}); ts_s = meta_m.get('timestamp_stats', {})
        if cat_s.get('unique_count',0)>0: recs.append(f"   - Use 'category' ({cat_s.get('unique_count','N/A')} unique).")
        if tag_s.get('unique_tag_count',0)>0: recs.append(f"   - Use 'tags' ({tag_s.get('unique_tag_count','N/A')} unique).")
        if up_s.get('count',0)>0: recs.append(f"   - Use 'upvotes' ({up_s.get('count',0):.0f} non-null).")
        if 'min_date' in ts_s : recs.append("   - Use 'timestamp' for recency.")
        if 'source' in self.df.columns and self.df['source'].notna().any(): recs.append("   - Use 'source' for attribution.")
        recs.extend(["5. Preprocess text for embedding.", "6. Consider Q augmentation.", 
                     "7. Iteratively evaluate RAG.", "8. Manually spot-check with sample CSV."])
        rag_text.extend([f"   - {r}" for r in recs])
        
        self._add_to_report_and_metrics("RAG Preparation Insights & Recommendations", "rag_recommendations",
                                        rag_text, metrics_data=metrics_rag)
        self._save_sample_for_review()

    def _save_sample_for_review(self):
        logger.info("Saving sample for review...")
        if self.df.empty: 
            logger.info("DataFrame empty, no sample CSV will be generated.")
            return

        sample_path = os.path.join(self.paths_cfg['output_dir'], self.paths_cfg['sample_csv'])
        cols_std = ['id', 'question_clean', 'answer_clean', 'category', 'qna_title', 'summary', 
                    'question_length_words', 'answer_length_words', 
                    'tags_normalized', 'upvotes', 'question_type_basic', 'source']
        existing_cols = [col for col in cols_std if col in self.df.columns]
        
        n_sample = min(100, len(self.df))
        if n_sample > 0:
            self.df[existing_cols].sample(n=n_sample, random_state=42).to_csv(sample_path, index=False, encoding='utf-8')
            logger.info(f"Sample data ({n_sample} records) saved to: {sample_path}")
        else:
            logger.info("No records to sample (DataFrame might be smaller than minimum sample size).")