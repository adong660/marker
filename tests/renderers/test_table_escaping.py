"""Model-free tests for table/form HTML ingest sanitization (upstream #881).

Cell content with raw "<" (e.g. "<LOQ") is eaten as a fake tag when external
HTML (OCR model / LLM / spreadsheet) is parsed. Fix: escape stray "<" at the
HTML-ingest boundaries (marker.util.escape_text_outside_tags), never at the
HTML-fragment joins (tablecell text_lines carry intentional HTML fragments).
"""

from bs4 import BeautifulSoup

from marker.processors.table import TableProcessor
from marker.providers.spreadsheet import SpreadSheetProvider
from marker.renderers.json import JSONRenderer
from marker.renderers.markdown import MarkdownRenderer
from marker.schema.blocks import Table, TableCell
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox
from marker.util import (
    HTML_FRAGMENT_ALLOWED_TAGS,
    escape_text_outside_tags,
    store_math_html,
)

TAGS = set(HTML_FRAGMENT_ALLOWED_TAGS)


def make_doc(*blocks) -> Document:
    page = PageGroup(
        polygon=PolygonBox.from_bbox((0, 0, 200, 200)),
        page_id=0,
        children=[],
        structure=[],
    )
    for block in blocks:
        page.add_full_block(block)
        page.add_structure(block)
    return Document(filepath="dummy.pdf", pages=[page])


def make_table_doc(*text_lines_lists) -> Document:
    page = PageGroup(
        polygon=PolygonBox.from_bbox((0, 0, 200, 200)),
        page_id=0,
        children=[],
        structure=[],
    )
    table = Table(polygon=PolygonBox.from_bbox((0, 0, 100, 100)), page_id=0)
    page.add_full_block(table)
    page.add_structure(table)
    for col_id, text_lines in enumerate(text_lines_lists):
        cell = TableCell(
            polygon=PolygonBox.from_bbox((0, col_id * 20, 50, (col_id + 1) * 20)),
            page_id=0,
            rowspan=1,
            colspan=1,
            row_id=0,
            col_id=col_id,
            is_header=False,
            text_lines=text_lines,
        )
        page.add_full_block(cell)
        table.add_structure(cell)
    return Document(filepath="dummy.pdf", pages=[page])


def leaf_htmls(json_output) -> list[str]:
    out = []

    def walk(node):
        children = getattr(node, "children", None)
        if not children:
            out.append(node.html)
            return
        for child in children:
            walk(child)

    for page in json_output.children:
        walk(page)
    return out


# --- sanitizer unit tests ---


def test_sanitize_escapes_stray_lt():
    assert escape_text_outside_tags("x <LOQ", TAGS) == "x &lt;LOQ"


def test_sanitize_escapes_lt_in_text():
    assert escape_text_outside_tags("a < b", TAGS) == "a &lt; b"


def test_sanitize_keeps_allowed_tags():
    s = 'a <i>x</i><br><b>y</b> <td colspan="2">c</td>'
    assert escape_text_outside_tags(s, TAGS) == s


def test_sanitize_math_island_like_store_math_html():
    s = "<math>y_{<l}</math>"
    assert escape_text_outside_tags(s, TAGS) == store_math_html(s) == (
        "<math>y_{&lt;l}</math>"
    )


def test_sanitize_escapes_malformed_tag_junk():
    junk = 'f(x) <l} \\mid=""'
    assert escape_text_outside_tags(junk, TAGS) == 'f(x) &lt;l} \\mid=""'


def test_sanitize_is_idempotent():
    sample = (
        '<table><tr><td>Val <LOQ & a < b</td><td><math>y_{<l}</math></td></tr></table>'
        "<i>ok</i> stray <div> and &amp; entity"
    )
    once = escape_text_outside_tags(sample, TAGS)
    assert escape_text_outside_tags(once, TAGS) == once


# --- contract guards (the upstream #997 trap): intentional HTML survives ---


def test_contract_intentional_html_in_text_lines():
    doc = make_table_doc(["54<i>.45</i>67<br>89<math>x</math>"])
    md = MarkdownRenderer({"extract_images": False})(doc).markdown
    assert "54 <i>.45</i> 67<br>89 $x$" in md


def test_contract_llm_table_style_cell():
    doc = make_table_doc(["Value 1 <math>x</math>"])
    md = MarkdownRenderer({"extract_images": False})(doc).markdown
    assert "Value 1 $x$" in md


# --- bug fix proof: plain text with stray "<" survives end to end ---


def test_table_cell_plain_lt_renders_complete():
    doc = make_table_doc(["&lt;LOQ"])  # post-sanitize fragment form
    md = MarkdownRenderer({"extract_images": False})(doc).markdown
    assert "<LOQ" in md  # decoded, complete, not eaten


def test_table_cells_complete_in_markdown_and_json():
    doc = make_table_doc(["a &lt; b"], ["&lt;LOQ"])
    md = MarkdownRenderer({"extract_images": False})(doc).markdown
    assert "a < b" in md
    assert "<LOQ" in md

    json_out = JSONRenderer({"extract_images": False})(doc)
    cell_htmls = leaf_htmls(json_out)
    joined = "".join(cell_htmls)
    # Table text keeps the sanitized entities in JSON (unlike the math A2 contract)
    assert "a &lt; b" in joined
    assert "&lt;LOQ" in joined
    assert "&amp;lt;" not in joined  # no double-escaping


# --- clean_table_html / clean_form_html (OCR model HTML ingest) ---


def test_clean_table_html_escapes_stray_lt():
    proc = object.__new__(TableProcessor)
    raw = "<table><tr><td>Val <LOQ</td></tr></table>"
    cleaned = proc.clean_table_html(raw)
    assert "Val &lt;LOQ" in cleaned
    assert BeautifulSoup(cleaned, "html.parser").find("td").text == "Val <LOQ"
    assert BeautifulSoup(cleaned, "html.parser").find("table") is not None


def test_clean_table_html_requires_table():
    proc = object.__new__(TableProcessor)
    assert proc.clean_table_html("<div>hi</div>") == ""
    assert proc.clean_table_html(None) == ""


def test_clean_table_html_rejects_repeat_loops():
    proc = object.__new__(TableProcessor)
    assert proc.clean_table_html("<table>" + "same phrase " * 400) == ""


def test_clean_form_html_escapes_stray_lt_no_table_required():
    proc = object.__new__(TableProcessor)
    cleaned = proc.clean_form_html("Field <LOQ")
    assert cleaned == "Field &lt;LOQ"
    assert proc.clean_form_html(None) == ""


# --- spreadsheet cell values ---


class _FakeCell:
    def __init__(self, value):
        self.value = value


class _FakeMerged:
    ranges = []


class _FakeSheet:
    def __init__(self, rows):
        self.rows = rows
        self.merged_cells = _FakeMerged()


def test_spreadsheet_escapes_cell_values():
    provider = object.__new__(SpreadSheetProvider)
    provider.temp_pdf_path = "/tmp/opencode/nonexistent-spreadsheet-test"
    sheet = _FakeSheet([[_FakeCell("<LOQ & <b>")]])
    html_out = provider._excel_to_html_table(sheet)
    assert "&lt;LOQ &amp; &lt;b&gt;" in html_out
    assert "<LOQ & <b>" not in html_out
