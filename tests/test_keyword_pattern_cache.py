"""Cache reuse must preserve each term's literal, language-aware scoring."""
import pytest

from puremacro.narrative.indices._kernels import _get_term_regex, count_keywords


@pytest.mark.parametrize('text,terms,language,negation,expected', [
    ('RISK risk risky', frozenset({'Risk', 'risk'}), 'en', False, 4),
    ('a.b a-b a+b c++ c++x', frozenset({'a.b', 'a+b', 'c++'}), 'en', False, 3),
    ('CRISIS económica crisis económicas', frozenset({'crisis', 'económica'}), 'es', False, 3),
    ('policy risk; policy risk', frozenset({'policy', 'policy risk', 'risk'}), 'en', False, 6),
    ('no risk now. one two three four five risk', frozenset({'risk'}), 'en', True, 1),
    ('不确定性 确定性', frozenset({'确定性'}), 'zh', False, 2),
    ('不确定性', frozenset({'确定性'}), 'zh', True, 0),
    ('リスク、リスク', frozenset({'リスク'}), 'ja', False, 2),
    ('リスクはない。リスク', frozenset({'リスク'}), 'ja', True, 1),
    ('', frozenset({'risk'}), 'en', True, 0),
])
def test_literal_keyword_counts(text, terms, language, negation, expected):
    assert count_keywords(text, terms, language, negation=negation) == expected


def test_pattern_cache_reused_for_negation_and_bounded():
    _get_term_regex.cache_clear()
    try:
        terms = frozenset({'policy', 'risk'})
        assert count_keywords('policy risk', terms) == 2
        assert count_keywords('no policy risk', terms, negation=True) == 0
        info = _get_term_regex.cache_info()
        assert (info.hits, info.misses, info.currsize) == (2, 2, 2)
        for i in range(1100):
            _get_term_regex(f'term{i}')
        assert _get_term_regex.cache_info().currsize == 1024
        before = _get_term_regex.cache_info()
        count_keywords('风险', frozenset({'风险'}), 'zh')
        assert _get_term_regex.cache_info() == before
    finally:
        _get_term_regex.cache_clear()
