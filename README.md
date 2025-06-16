# Q&A Dataset Analyzer for RAG Preparation

## 1. Overview

The **Q&A Dataset Analyzer** is a Python tool for comprehensive Exploratory Data Analysis (EDA) of question-answer datasets. It assesses data quality, suitability for Retrieval Augmented Generation (RAG) systems, and provides preprocessing recommendations, crucial for preparing data for AI/NLP applications.

## 2. Features

- Parses nested JSON Q&A structures.
- Performs HTML cleaning and text normalization.
- Analyzes content quality (missing values, lengths, short content).
- Detects exact and near-duplicate questions/answers.
- Analyzes metadata (categories, tags, timestamps, upvotes).
- Basic regex-based question type classification.
- Generates RAG-specific recommendations.
- Highly configurable via YAML.
- Outputs HTML, Text, and JSON metrics reports, plus a data sample CSV.
- Modular and extensible codebase.

## 3. Project Structure

```
qa_analyzer_project/
├── main_analyzer.py # Main execution script
├── config/
│   ├── default_config.yaml # Default settings
│   └── report_template.html # HTML report template
├── qa_toolkit/ # Core library modules
│   ├── __init__.py
│   ├── config_loader.py
│   ├── data_loader.py
│   ├── text_processor.py
│   ├── analysis_engine.py
│   ├── reporting.py
│   └── utils.py
└── requirements.txt # Python dependencies
```

## 4. Prerequisites

- Python 3.7+
- `pip` (Python package installer)

## 5. Setup and Installation

1.  **Clone or Download:**

    ```bash
    # If using Git:
    # git clone https://github.com/Ibnuawf/Q-A-Dataset-Analyzer-for-RAG-Preparation
    # cd qa_analyzer_project
    ```

    Alternatively, download the project files and navigate to the `qa_analyzer_project/` directory.

2.  **Create Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 6. Configuration

The analyzer is configured using YAML files.

- **Default Configuration:** `config/default_config.yaml` contains all default settings.
- **Custom Configuration:** Create your own `.yaml` file (e.g., `my_config.yaml`) to override defaults.

**Key `column_mapping` (in your config file - VERY IMPORTANT):**
This section maps the script's internal standardized field names to the actual field names found in your JSON data after initial parsing by `pandas.json_normalize()`. **You MUST configure this correctly for your specific dataset.**

_Example for the `islamqa.info` structure previously discussed:_

```yaml
# In your custom_config.yaml or by editing default_config.yaml
column_mapping:
  id: 'url'                        # Standardized 'id' will come from the 'url' field of each Q&A item
  question: 'question'             # Standardized 'question' from 'question' field
  answer: 'answer'                 # Standardized 'answer' from 'answer' field
  category: 'parent_category_title'# Standardized 'category' from the propagated parent topic title
  source: 'url'                    # Standardized 'source' from 'url' field
  qna_title: 'title'               # Standardized 'qna_title' from the 'title' field of each Q&A item
  summary: 'summary'               # Standardized 'summary' from the 'summary' field
  # Map 'tags', 'timestamp', 'upvotes' if these fields exist within your individual Q&A items
```

Refer to config/default_config.yaml for all available options (file paths, analysis parameters, steps to run, reporting flags).

## 7. How to Run

Navigate to the qa_analyzer_project/ directory in your terminal.

With default configuration:

```bash
python main_analyzer.py /path/to/your/qna_dataset.json
```

With a custom configuration file:

```bash
python main_analyzer.py /path/to/your/qna_dataset.json --config /path/to/your_custom_config.yaml
```

## 8. Output Reports

Generated in the configured output directory (default: qa_analysis_reports_pro/):

- **HTML Report (.html):** Detailed, visual report.
- **Text Report (.txt):** Plain text summary.
- **JSON Metrics (.json):** Key statistics for programmatic use.
- **Sample CSV (.csv):** Data sample for manual review.
- **Log File (.log):** Execution logs for debugging.
- **Plot Images (plot_images/):** Individual plot files (if configured).

## 9. Troubleshooting

- **ModuleNotFoundError:** Ensure dependencies are installed (see Step 5).
- **FileNotFoundError:** Verify paths to input/config files.
- **ValueError: Missing essential standardized columns...:** This is a column mapping issue. Carefully check and correct the column_mapping section in your configuration (see Step 6) to match your JSON data's actual field names.
- **Unicode Errors:** Log files are UTF-8. Console display depends on your terminal settings.
