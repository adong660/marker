"""Model-free tests for math-payload escaping (the "fake tag" truncation bug).

Latex with bare "<" (e.g. y_{<l}) embedded raw into HTML is eaten as a fake tag
by BeautifulSoup and formulas render truncated. Generation stores the <math>
payload escaped (marker.util.store_math_html); JSON extraction un-escapes it
back to raw latex inside <math> (A2 contract, cf. data/examples/json/switch_trans.json).
"""

from bs4 import BeautifulSoup

from marker.output import json_to_html
from marker.renderers.html import HTMLRenderer
from marker.renderers.json import JSONRenderer
from marker.renderers.markdown import MarkdownRenderer
from marker.schema.blocks import Equation, Text
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox
from marker.util import escape_math_payload, store_math_html, unescape_math_payload

NASTY_LATEX = r"y_{<l}, a < b, x > y, A & B, \text{don't}"


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


def make_equation(latex: str) -> Equation:
    # Stored the way the pipeline stores it: escaped math payload at generation.
    return Equation(
        polygon=PolygonBox.from_bbox((0, 0, 50, 20)),
        page_id=0,
        html=store_math_html(f'<math display="block">{latex}</math>'),
    )


def test_escape_math_payload_scope_and_quotes():
    out = escape_math_payload(f"<math>{NASTY_LATEX}</math><p>a < b</p>")
    assert out.startswith("<math>")
    assert out.endswith("<p>a < b</p>")  # outside <math> untouched
    assert "y_{&lt;l}" in out
    assert "don't" in out  # quote=False: quotes left alone


def test_escape_math_payload_double_application_is_detectable():
    once = escape_math_payload(f"<math>{NASTY_LATEX}</math>")
    twice = escape_math_payload(once)
    assert "&amp;lt;" not in once
    assert "&amp;lt;" in twice  # documented non-idempotence: apply once at boundaries


def test_unescape_math_payload_is_exact_inverse():
    s = f'<math display="block">{NASTY_LATEX}</math><p>a &amp; b</p>'
    assert unescape_math_payload(escape_math_payload(s)) == s


def test_store_math_html_canonical_and_idempotent():
    raw = f"<math>{NASTY_LATEX}</math>"
    escaped = escape_math_payload(raw)
    assert store_math_html(raw) == store_math_html(escaped) == escaped
    assert store_math_html(store_math_html(raw)) == escaped


def test_store_math_html_preserves_intentional_html_outside_math():
    s = "<p>see <i>it</i> and <b>bold</b> &amp; more</p>"
    assert store_math_html(s) == s


def test_markdown_math_not_truncated_raw_characters():
    doc = make_doc(make_equation(NASTY_LATEX))
    md = MarkdownRenderer({"extract_images": False})(doc).markdown
    assert f"$${NASTY_LATEX}$$" in md  # complete, raw characters
    assert "&amp;lt;" not in md  # double-escape guard
    assert "&lt;" not in md  # no entity leftovers in markdown


def test_json_math_carries_raw_latex():
    doc = make_doc(make_equation(NASTY_LATEX))
    output = JSONRenderer({"extract_images": False})(doc)
    eq_html = output.children[0].children[0].html
    assert f'<math display="block">{NASTY_LATEX}</math>' in eq_html  # A2 raw contract
    assert "y_{<l}" in eq_html  # non-truncated past the fake-tag point
    assert "&amp;lt;" not in eq_html  # double-escape guard


def test_json_to_html_reescapes_before_reparsing():
    doc = make_doc(make_equation(NASTY_LATEX))
    output = JSONRenderer({"extract_images": False})(doc)
    eq = output.children[0].children[0]
    html_str = json_to_html(eq)  # JSON holds raw latex; re-escape for html use
    assert "y_{&lt;l}" in html_str
    assert "y_{<l}" not in html_str
    math_text = BeautifulSoup(html_str, "html.parser").find("math").text
    assert NASTY_LATEX in math_text  # parse round-trip is lossless


def test_intentional_html_outside_math_preserved():
    doc = make_doc(
        Text(
            polygon=PolygonBox.from_bbox((0, 30, 50, 50)),
            page_id=0,
            html="<p>see <i>it</i> and <b>bold</b> here</p>",
        )
    )
    # JSON/HTML keep the tags verbatim
    output = JSONRenderer({"extract_images": False})(doc)
    assert output.children[0].children[0].html == (
        "<p>see <i>it</i> and <b>bold</b> here</p>"
    )
    html_out = HTMLRenderer({"extract_images": False})(doc).html
    assert "<i>" in html_out and "<b>" in html_out
    assert "&lt;i&gt;" not in html_out
    # Markdownify converts the tags to emphasis (pre-existing renderer behavior)
    md = MarkdownRenderer({"extract_images": False})(doc).markdown
    assert "see *it* and **bold** here" in md
