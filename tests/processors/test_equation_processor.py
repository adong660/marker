import pytest

from marker.schema import BlockTypes
from marker.processors.equation import EquationProcessor


@pytest.mark.config({"page_range": [0]})
def test_equation_processor(pdf_document, recognition_model):
    processor = EquationProcessor(recognition_model)
    processor(pdf_document)

    for block in pdf_document.pages[0].children:
        if block.removed:
            # Old layout blocks retired by full-page OCR structure replacement
            continue
        if block.block_type == BlockTypes.Equation:
            assert block.html is not None


# --- fix_latex unit tests (model-free: fix_latex does not use self) ---


def _fix_latex(math_html: str) -> str:
    processor = object.__new__(EquationProcessor)
    return processor.fix_latex(math_html)


def test_fix_latex_escapes_raw_lt():
    assert _fix_latex("<math>y_{<l}</math>") == '<math display="block">y_{&lt;l}</math>'


def test_fix_latex_escaped_and_raw_inputs_identical():
    escaped = "<math>y_{&lt;l}</math>"
    raw = "<math>y_{<l}</math>"
    assert _fix_latex(escaped) == _fix_latex(raw) == '<math display="block">y_{&lt;l}</math>'


def test_fix_latex_model_junk_with_stray_lt_not_corrupted():
    assert _fix_latex("<math>a < b</math>") == '<math display="block">a &lt; b</math>'
    assert _fix_latex(r"<math>x_{<l} \cdot 2</math>") == (
        '<math display="block">x_{&lt;l} \\cdot 2</math>'
    )


def test_fix_latex_p_unwrap_and_display_block():
    assert _fix_latex("<p><math>x</math></p>") == '<math display="block">x</math>'
    assert _fix_latex('<math display="inline">x</math>') == (
        '<math display="block">x</math>'
    )


def test_fix_latex_newline_cleanup_preserved():
    # Historical cleanup: literal \n junk at payload edges (kept before a letter)
    assert _fix_latex(r"<math>\n5</math>") == '<math display="block">5</math>'
    assert _fix_latex(r"<math>x\n</math>") == '<math display="block">x</math>'


def test_fix_latex_no_math_returns_empty():
    assert _fix_latex("just text") == ""
    assert _fix_latex("") == ""
