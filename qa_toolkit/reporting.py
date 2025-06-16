# qa_toolkit/reporting.py
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any 
from jinja2 import Environment, FileSystemLoader, select_autoescape
import numpy as np  # <--- ADD THIS IMPORT
import pandas as pd # <--- ADD THIS IMPORT for pd.Timestamp

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, config: Dict, analysis_results: List[Dict[str, Any]], metrics_data: Dict[str, Any]):
        # ... (rest of the constructor) ...
        self.config = config # Ensure self.config is assigned
        self.paths_cfg = config['file_paths']
        self.report_cfg = config['reporting']
        self.analysis_results_html_parts = analysis_results
        self.metrics_data = metrics_data
        self.report_title = f"Q&A Dataset Analysis Report (Pro) - {os.path.basename(self.paths_cfg['input_dataset'])}"


    def _compile_text_report_content(self) -> str:
        # ... (rest of the method) ...
        text_lines = []
        for section in self.analysis_results_html_parts:
            text_lines.append(f"\n--- {section['title'].upper()} ---\n")
            content = section.get('text_content')
            if section['type'] == 'html_table':
                text_lines.append(f"[HTML Table for {section['title']} - view in HTML report]")
            elif section['type'] == 'plot':
                text_lines.append(f"[Plot for {section['title']} - view in HTML report or image file: {section.get('plot_uri', 'N/A')}]")
            elif content:
                if isinstance(content, list):
                    for item in content: text_lines.append(str(item))
                else:
                    text_lines.append(str(content))
            text_lines.append("\n" + "=" * 70)
        return "".join(text_lines)

    def generate_all_reports(self):
        logger.info("Finalizing reports and outputs...")

        # Save Text Report
        if self.report_cfg['save_text_report']:
            # ... (text report saving logic) ...
            text_report_header = f"{self.report_title}\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nSource File: {os.path.basename(self.paths_cfg['input_dataset'])}\n" + "=" * 70 + "\n"
            full_text_report = text_report_header + self._compile_text_report_content()
            text_report_path = os.path.join(self.paths_cfg['output_dir'], self.paths_cfg['report_text'])
            try:
                with open(text_report_path, 'w', encoding='utf-8') as f:
                    f.write(full_text_report)
                logger.info(f"Text report saved to: {text_report_path}")
            except Exception as e:
                logger.error(f"Failed to save text report: {e}")


        # Save JSON Metrics
        if self.report_cfg['save_json_metrics']:
            json_metrics_path = os.path.join(self.paths_cfg['output_dir'], self.paths_cfg['report_json_metrics'])
            try:
                def default_serializer(o): # This is line 58 or so
                    if isinstance(o, (np.integer, np.int64)): return int(o)      # Line 59
                    if isinstance(o, (np.floating, np.float64)): return float(o)  # Line 60
                    if isinstance(o, np.bool_): return bool(o)                   # Line 61
                    if isinstance(o, pd.Timestamp): return o.isoformat()
                    if isinstance(o, (set, frozenset)): return list(o)
                    # Fallback for other non-serializable types
                    if hasattr(o, 'to_dict') and callable(o.to_dict): return o.to_dict()
                    if hasattr(o, '__dict__'): return o.__dict__
                    return str(o)


                serializable_metrics = json.loads(json.dumps(self.metrics_data, default=default_serializer))
                with open(json_metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_metrics, f, indent=4)
                logger.info(f"JSON metrics saved to: {json_metrics_path}")
            except Exception as e:
                logger.error(f"Failed to save JSON metrics: {e}", exc_info=True)

        # Generate HTML Report
        if self.report_cfg['generate_html_report']:
            # ... (HTML report generation logic from previous correct version) ...
            html_report_path = os.path.join(self.paths_cfg['output_dir'], self.paths_cfg['report_html'])
            try:
                script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
                template_dir = os.path.join(script_dir, 'config')
                template_filename = 'report_template.html'
                
                if os.path.exists(os.path.join(template_dir, template_filename)):
                    env = Environment(loader=FileSystemLoader(template_dir), autoescape=select_autoescape(['html', 'xml']))
                    template = env.get_template(template_filename)
                else:
                    logger.error(f"HTML template file '{template_filename}' not found in '{template_dir}'. Cannot generate HTML report.")
                    return

                html_output = template.render(
                    report_title=self.report_title,
                    generation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    source_file=os.path.basename(self.paths_cfg['input_dataset']),
                    sections=self.analysis_results_html_parts
                )
                with open(html_report_path, 'w', encoding='utf-8') as f:
                    f.write(html_output)
                logger.info(f"HTML report saved to: {html_report_path}")
            except Exception as e:
                logger.error(f"Failed to generate HTML report: {e}", exc_info=True)