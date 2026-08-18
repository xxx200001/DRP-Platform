"""OCR 结构化引擎（规范 4.1）：归一化词典 + 化验单文本解析。"""

from .lexicon import IndicatorLexicon, LexiconMatch, fold_confusions
from .parser import (
    LabReportParser,
    ParsedRow,
    ParseReport,
    parse_lab_text,
    rows_to_frame,
)

__all__ = [
    "IndicatorLexicon",
    "LexiconMatch",
    "fold_confusions",
    "LabReportParser",
    "ParsedRow",
    "ParseReport",
    "parse_lab_text",
    "rows_to_frame",
]
