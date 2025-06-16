# qa_toolkit/text_processor.py
import re
import html
from bs4 import BeautifulSoup
from typing import Any

class TextProcessor:
    @staticmethod
    def clean_html_content(text: Any) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        try:
            # More robust unescaping and cleaning
            decoded_text = html.unescape(text)
            soup = BeautifulSoup(decoded_text, 'html.parser')
            
            # Remove script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            
            # Get text, ensuring spaces between elements, then strip and normalize whitespace
            text_content = soup.get_text(separator=' ', strip=True)
            return re.sub(r'\s+', ' ', text_content).strip()
        except Exception:  # Fallback if BeautifulSoup fails
            return re.sub(r'<[^>]+>', '', str(text)).strip()

    @staticmethod
    def normalize_text_for_comparison(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Collapse multiple whitespaces
        return text.strip()

    @staticmethod
    def calculate_jaccard_similarity(text1: str, text2: str) -> float:
        words1 = set(TextProcessor.normalize_text_for_comparison(text1).split())
        words2 = set(TextProcessor.normalize_text_for_comparison(text2).split())

        if not words1 and not words2: return 1.0  # Both empty strings are identical
        if not words1 or not words2: return 0.0   # One empty, one not

        intersection_len = len(words1.intersection(words2))
        union_len = len(words1.union(words2))
        return intersection_len / union_len if union_len > 0 else 0.0

    @staticmethod
    def classify_question_type_basic(question_text: str) -> str:
        if not isinstance(question_text, str): return "unknown"
        q_lower = question_text.lower().strip()
        if not q_lower: return "empty"

        # Order matters for some of these regexes
        if re.match(r"^(is|are|was|were|do|does|did|can|could|will|would|should|may|might|has|have|had)\b", q_lower):
            return "yes/no/confirmation" if ' or ' not in q_lower else "choice"
        if re.match(r"^(what|which|who|whom|whose)\b", q_lower): return "what/which/who"
        if re.match(r"^(where)\b", q_lower): return "where"
        if re.match(r"^(when)\b", q_lower): return "when"
        if re.match(r"^(why)\b", q_lower): return "why"
        if re.match(r"^(how)\b", q_lower): return "how"
        if re.match(r"^(list|define|explain|describe|compare|contrast|name|give|tell me about)\b", q_lower): return "descriptive/command"
        
        if '?' in q_lower: return "other_interrogative"
        return "statement/unknown"