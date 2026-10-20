"""Builds the manuscript as a Word file from plain-text sources.

    python code/paper/build_docx.py --out paper/CAG-IDS_Final.docx --pdf

Sources (all in this folder unless noted):
  content.md            the text, in the light markup described below
  references.json       verified references; only the cited ones are printed,
                        numbered in order of first citation
  tables_static.json    tables that describe the design (written by hand)
  generated/            tables_generated.json, values.json and figures/, all
                        written by scripts/analyze.py from the experiment outputs
  equations/            the display equations (Office Math XML)
  template/             styles, numbering, footer, theme and settings of the
                        manuscript (single column, A4, Times New Roman 11 pt,
                        1.15 line spacing)

Markup
  @title: / @authors: / @keywords:      front matter
  @abstract:                            the following paragraph is the abstract
  # 1. Heading {#sec:label}             heading levels 1 to 3
  - item                                bullet item
  @eq: name {#eq:label}                 numbered display equation
  @table: key                           caption, table and note
  @fig: file | width in inches | caption {#fig:label}
  @algorithm: name                      verbatim block from equations/
  @references                           the reference list
  **bold**  *italic*  ***bold italic***  _{subscript}  ^{superscript}
  [@key] / [@key1, @key2]               citations
  [sec:label] [tab:key] [fig:label] [eq:label]   cross-references
  {{key}}                               a value from generated/values.json
"""
import argparse
import datetime
import json
import os
import re
import struct
import subprocess
import zipfile
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
TEXT_WIDTH = 9026  # dxa, A4 with 1-inch margins
EMU_PER_INCH = 914400


# ----------------------------------------------------------------------------
# inline markup

TOKEN = re.compile(r"(\*\*\*.+?\*\*\*|\*\*.+?\*\*|\*.+?\*|_\{.+?\}|\^\{.+?\})")


def rpr(bold=False, italic=False, size=None, va=None):
    parts = []
    if bold:
        parts.append("<w:b/><w:bCs/>")
    if italic:
        parts.append("<w:i/><w:iCs/>")
    if size:
        parts.append('<w:sz w:val="{0}"/><w:szCs w:val="{0}"/>'.format(size))
    if va:
        parts.append('<w:vertAlign w:val="{}"/>'.format(va))
    return "<w:rPr>{}</w:rPr>".format("".join(parts)) if parts else ""


def run(text, bold=False, italic=False, size=None, va=None):
    if text == "":
        return ""
    out = []
    pieces = text.split("\t")
    for i, piece in enumerate(pieces):
        if i > 0:
            out.append("<w:r>{}<w:tab/></w:r>".format(rpr(bold, italic, size, va)))
        if piece:
            out.append('<w:r>{}<w:t xml:space="preserve">{}</w:t></w:r>'.format(
                rpr(bold, italic, size, va), escape(piece)))
    return "".join(out)


def inline(text, size=None, bold=False, italic=False):
    """Converts the inline markup of one paragraph or cell into runs."""
    out = []
    pos = 0
    for m in TOKEN.finditer(text):
        if m.start() > pos:
            out.append(run(text[pos:m.start()], bold, italic, size))
        tok = m.group(0)
        if tok.startswith("***"):
            out.append(inline(tok[3:-3], size, True, True))
        elif tok.startswith("**"):
            out.append(inline(tok[2:-2], size, True, italic))
        elif tok.startswith("*"):
            out.append(inline(tok[1:-1], size, bold, True))
        elif tok.startswith("_{"):
            out.append(sub_sup(tok[2:-1], size, bold, italic, "subscript"))
        elif tok.startswith("^{"):
            out.append(sub_sup(tok[2:-1], size, bold, italic, "superscript"))
        pos = m.end()
    if pos < len(text):
        out.append(run(text[pos:], bold, italic, size))
    return "".join(out)


def sub_sup(text, size, bold, italic, va):
    # sub/superscripts inherit the italic of the symbol unless marked up inside
    out = []
    pos = 0
    for m in re.finditer(r"\*\*.+?\*\*|\*.+?\*", text):
        if m.start() > pos:
            out.append(run(text[pos:m.start()], bold, italic, size, va))
        tok = m.group(0)
        if tok.startswith("**"):
            out.append(run(tok[2:-2], True, italic, size, va))
        else:
            out.append(run(tok[1:-1], bold, True, size, va))
        pos = m.end()
    if pos < len(text):
        out.append(run(text[pos:], bold, italic, size, va))
    return "".join(out)


def para(runs, spacing='w:after="60"', jc="both", indent=None, extra_ppr="", style=None):
    ppr = []
    if style:
        ppr.append('<w:pStyle w:val="{}"/>'.format(style))
    ppr.append(extra_ppr)
    ppr.append("<w:spacing {}/>".format(spacing))
    if indent:
        ppr.append(indent)
    if jc:
        ppr.append('<w:jc w:val="{}"/>'.format(jc))
    return "<w:p><w:pPr>{}</w:pPr>{}</w:p>".format("".join(ppr), runs)


# ----------------------------------------------------------------------------
# document parts

class Builder:
    def __init__(self, content_path, tables, values, references, gen_dir, draft=False):
        self.draft = draft
        self.tables = tables
        self.values = values
        self.references = references
        self.gen_dir = gen_dir
        self.lines = open(content_path, encoding="utf-8").read().splitlines()
        self.cite_order = []
        self.labels = {}
        self.images = []
        self.missing = set()

    # first pass: numbers for sections, tables, figures, equations and citations
    def number(self):
        n_tab = n_fig = n_eq = 0
        for line in self.lines:
            s = line.strip()
            m = re.match(r"^(#{1,3})\s+(.*?)\s*(\{#(sec:[\w-]+)\})?$", s)
            if m and m.group(4):
                num = re.match(r"^([\d.]+)", m.group(2))
                self.labels[m.group(4)] = num.group(1).rstrip(".") if num else m.group(2)
            if s.startswith("@table:"):
                n_tab += 1
                self.labels["tab:" + s.split(":", 1)[1].strip()] = str(n_tab)
            elif s.startswith("@fig:"):
                n_fig += 1
                lab = re.search(r"\{#(fig:[\w-]+)\}", s)
                if lab:
                    self.labels[lab.group(1)] = str(n_fig)
            elif s.startswith("@eq:"):
                n_eq += 1
                lab = re.search(r"\{#(eq:[\w-]+)\}", s)
                if lab:
                    self.labels[lab.group(1)] = str(n_eq)
        # citations in order of first appearance in the document, tables included
        for line in self.lines:
            s = line.strip()
            if s.startswith("%%"):
                continue
            if s.startswith("@table:"):
                s = json.dumps(self.tables.get(s.split(":", 1)[1].strip(), {}), ensure_ascii=False)
            for group in re.findall(r"\[@[^\]]+\]", s):
                for key in re.findall(r"@([\w-]+)", group):
                    if key not in self.cite_order:
                        self.cite_order.append(key)
        unknown = [k for k in self.cite_order if k not in self.references]
        if unknown:
            raise SystemExit("unknown citation keys: " + ", ".join(unknown))

    def resolve(self, text):
        def value(m):
            key = m.group(1)
            if key not in self.values:
                self.missing.add(key)
                return "??" + key + "??"
            return str(self.values[key])
        text = re.sub(r"\{\{([\w-]+)\}\}", value, text)

        def cite(m):
            keys = re.findall(r"@([\w-]+)", m.group(0))
            nums = sorted(self.cite_order.index(k) + 1 for k in keys)
            groups, start, prev = [], nums[0], nums[0]
            for n in nums[1:] + [None]:
                if n is not None and n == prev + 1:
                    prev = n
                    continue
                groups.append(str(start) if start == prev else ("{}, {}".format(start, prev) if prev == start + 1
                                                                 else "{}-{}".format(start, prev)))
                if n is not None:
                    start = prev = n
            return "[" + ", ".join(groups) + "]"
        text = re.sub(r"\[@[^\]]+\]", cite, text)

        def xref(m):
            kind, key = m.group(1), m.group(2)
            label = "{}:{}".format(kind, key)
            if label not in self.labels:
                if self.draft:
                    return "??" + label + "??"
                raise SystemExit("unknown cross-reference " + label)
            num = self.labels[label]
            return {"sec": "Section " + num, "tab": "Table " + num, "fig": "Fig. " + num, "eq": "(" + num + ")"}[kind]
        text = re.sub(r"\[(sec|tab|fig|eq):([\w-]+)\]", xref, text)
        # bare numbers for cross-references when the word is already written
        text = re.sub(r"\[#(sec|tab|fig|eq):([\w-]+)\]",
                      lambda m: self.labels.get("{}:{}".format(m.group(1), m.group(2)), "??") if self.draft
                      else self.labels["{}:{}".format(m.group(1), m.group(2))], text)
        return text

    # ---------------------------------------------------------------- blocks
    def heading(self, level, text):
        spacing = {1: 'w:before="260" w:after="100"', 2: 'w:before="180" w:after="60"', 3: 'w:before="140" w:after="60"'}[level]
        return ('<w:p><w:pPr><w:pStyle w:val="Heading{}"/><w:keepNext/><w:spacing {}/></w:pPr>{}</w:p>'
                .format(level, spacing, inline(self.resolve(text))))

    def equation(self, name, number):
        omml = open(os.path.join(HERE, "equations", name + ".xml"), encoding="utf-8").read()
        return ('<w:p><w:pPr><w:tabs><w:tab w:val="center" w:pos="4513"/><w:tab w:val="right" w:pos="9026"/></w:tabs>'
                '<w:spacing w:before="100" w:after="100"/></w:pPr><w:r><w:tab/></w:r>{}<w:r><w:tab/><w:t>({})</w:t></w:r></w:p>'
                .format(omml, number))

    def caption(self, kind, number, text, before=True):
        size = 20
        runs = run("{} {}. ".format(kind, number), bold=True, size=size) + inline(self.resolve(text), size=size)
        if kind == "Table":
            return para(runs, spacing='w:before="180" w:after="60" w:line="240" w:lineRule="auto"', jc=None,
                        extra_ppr="<w:keepNext/>")
        return para(runs, spacing='w:after="160" w:line="240" w:lineRule="auto"', jc="both")

    def note(self, text):
        runs = run("Note. ", italic=True, size=17) + inline(self.resolve(text), size=17)
        return para(runs, spacing='w:before="40" w:after="160" w:line="230" w:lineRule="auto"', jc="both")

    def table(self, key):
        spec = self.tables.get(key)
        if spec is None:
            if self.draft:
                return para(run("[table {} not generated yet]".format(key), italic=True))
            raise SystemExit("table not found: " + key)
        number = self.labels["tab:" + key]
        out = [self.caption("Table", number, spec["caption"])]
        widths = spec["colWidths"]
        total = float(sum(widths))
        dxa = [int(round(TEXT_WIDTH * w / total)) for w in widths]
        dxa[-1] = TEXT_WIDTH - sum(dxa[:-1])
        size = int(spec.get("fontSize", 17))
        align = spec.get("align", ["left"] * len(widths))
        mid = set(spec.get("midRules", []))
        rows = []
        header = spec.get("header", [])
        body = spec.get("rows", [])

        # tables of up to 16 rows are kept on one page together with their note
        keep_together = spec.get("keepTogether", len(body) <= 16)

        def cell(content, width, col, is_header, top, bottom, span=1):
            if isinstance(content, dict):
                text = content.get("text", "")
                bold = content.get("bold", False)
                cell_align = content.get("align", align[col] if col < len(align) else "left")
                span = content.get("colspan", span)
            else:
                text, bold, cell_align = str(content), False, align[col] if col < len(align) else "left"
            if is_header:
                bold = True
                cell_align = align[col] if col < len(align) and col > 0 else "left"
            text = self.resolve(text)
            borders = ('<w:tcBorders><w:top w:val="single" w:sz="{}" w:space="0" w:color="000000"/>'
                       '<w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
                       '<w:bottom w:val="single" w:sz="{}" w:space="0" w:color="000000"/>'
                       '<w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/></w:tcBorders>').format(top, bottom)
            grid = '<w:gridSpan w:val="{}"/>'.format(span) if span > 1 else ""
            ppr = "<w:pPr>{}{}</w:pPr>".format("<w:keepNext/>" if (is_header or keep_together) else "",
                                               '<w:jc w:val="{}"/>'.format(cell_align) if cell_align != "left" else "")
            paras = []
            for piece in text.split("\n"):
                paras.append("<w:p>{}{}</w:p>".format(ppr, inline(piece, size=size, bold=bold)))
            return ('<w:tc><w:tcPr><w:tcW w:w="{}" w:type="dxa"/>{}{}<w:tcMar><w:top w:w="22" w:type="dxa"/>'
                    '<w:left w:w="55" w:type="dxa"/><w:bottom w:w="22" w:type="dxa"/><w:right w:w="55" w:type="dxa"/>'
                    '</w:tcMar><w:vAlign w:val="center"/></w:tcPr>{}</w:tc>').format(width, grid, borders, "".join(paras))

        def row(cells, is_header, index):
            parts = []
            col = 0
            top = 8 if (not is_header and (index == 0 or (index - 1) in mid)) else 4
            bottom = 8 if is_header else (8 if index in mid else 4)
            for content in cells:
                span = content.get("colspan", 1) if isinstance(content, dict) else 1
                width = sum(dxa[col:col + span])
                parts.append(cell(content, width, col, is_header, top, bottom, span))
                col += span
            trpr = "<w:trPr><w:cantSplit/>{}<w:jc w:val=\"center\"/></w:trPr>".format("<w:tblHeader/>" if is_header else "")
            return ('<w:tr><w:tblPrEx><w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:bottom w:w="0" w:type="dxa"/>'
                    '</w:tblCellMar></w:tblPrEx>{}{}</w:tr>').format(trpr, "".join(parts))

        for h in header:
            rows.append(row(h, True, -1))
        for i, r in enumerate(body):
            rows.append(row(r, False, i))
        grid = "".join('<w:gridCol w:w="{}"/>'.format(w) for w in dxa)
        out.append(
            '<w:tbl><w:tblPr><w:tblW w:w="{}" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>'
            '<w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
            '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
            '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
            '</w:tblBorders><w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="10" w:type="dxa"/><w:right w:w="10" w:type="dxa"/>'
            '</w:tblCellMar><w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/>'
            '</w:tblPr><w:tblGrid>{}</w:tblGrid>{}</w:tbl>'.format(TEXT_WIDTH, grid, "".join(rows)))
        if spec.get("notes"):
            out.append(self.note(spec["notes"]))
        else:
            out.append('<w:p><w:pPr><w:spacing w:before="0" w:after="120"/></w:pPr></w:p>')
        return "".join(out)

    def figure(self, spec):
        parts = [p.strip() for p in spec.split("|")]
        path, width_in, caption = parts[0], float(parts[1]), parts[2]
        label = re.search(r"\{#(fig:[\w-]+)\}", caption)
        caption = re.sub(r"\s*\{#fig:[\w-]+\}", "", caption)
        number = self.labels[label.group(1)] if label else "?"
        full = path if os.path.isabs(path) else os.path.join(self.gen_dir, path)
        if not os.path.exists(full):
            full = os.path.join(HERE, path)
        if not os.path.exists(full) and self.draft:
            return para(run("[figure {} not generated yet]".format(path), italic=True))
        w_px, h_px = png_size(full)
        cx = int(width_in * EMU_PER_INCH)
        cy = int(cx * h_px / w_px)
        rid = "rId{}".format(100 + len(self.images))
        self.images.append((rid, full))
        n = len(self.images)
        drawing = (
            '<w:r><w:rPr><w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            '<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="{n}" name="Picture {n}"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>'
            '</wp:cNvGraphicFramePr><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{n}" name="image{n}.png"/>'
            '<pic:cNvPicPr><a:picLocks noChangeAspect="1" noChangeArrowheads="1"/></pic:cNvPicPr></pic:nvPicPr>'
            '<pic:blipFill><a:blip r:embed="{rid}"/><a:srcRect/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            '<pic:spPr bwMode="auto"><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>'
        ).format(cx=cx, cy=cy, n=n, rid=rid)
        img = '<w:p><w:pPr><w:keepNext/><w:spacing w:before="160" w:after="60"/><w:jc w:val="center"/></w:pPr>{}</w:p>'.format(drawing)
        return img + self.caption("Fig.", number, caption)

    def reference_list(self):
        out = []
        for i, key in enumerate(self.cite_order, start=1):
            runs = run("[{}]".format(i), size=18) + run("\t", size=18) + inline(self.references[key], size=18)
            out.append('<w:p><w:pPr><w:tabs><w:tab w:val="left" w:pos="460"/></w:tabs>'
                       '<w:spacing w:after="30" w:line="228" w:lineRule="auto"/><w:ind w:left="460" w:hanging="460"/></w:pPr>{}</w:p>'
                       .format(runs))
        return "".join(out)

    # ---------------------------------------------------------------- body
    def body(self):
        self.number()
        out = []
        buffer = []
        prev = "heading"
        state = {"abstract": False}
        eq_count = [0]

        def flush():
            nonlocal prev
            if not buffer:
                return
            text = self.resolve(" ".join(buffer))
            buffer.clear()
            if state["abstract"]:
                out.append(para(inline(text), spacing='w:after="120"'))
                state["abstract"] = False
                prev = "abstract"
                return
            indent = '<w:ind w:firstLine="360"/>' if prev == "para" else None
            out.append(para(inline(text), indent=indent))
            prev = "para"

        for raw in self.lines:
            line = raw.rstrip()
            s = line.strip()
            if s.startswith("%%"):
                continue
            if not s:
                flush()
                continue
            if s.startswith("@title:"):
                out.append('<w:p><w:pPr><w:spacing w:after="240" w:line="300" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>{}</w:p>'
                           .format(run(s[7:].strip(), bold=True, size=34)))
                continue
            if s.startswith("@authors:"):
                out.append('<w:p><w:pPr><w:spacing w:after="80"/><w:jc w:val="center"/></w:pPr>{}</w:p>'
                           .format(run(s[9:].strip(), size=24)))
                continue
            if s.startswith("@abstract:"):
                flush()
                out.append('<w:p><w:pPr><w:pBdr><w:top w:val="single" w:sz="6" w:space="8" w:color="000000"/></w:pBdr>'
                           '<w:spacing w:before="120" w:after="80"/></w:pPr>{}</w:p>'.format(run("Abstract", bold=True)))
                state["abstract"] = True
                continue
            if s.startswith("@keywords:"):
                flush()
                out.append('<w:p><w:pPr><w:pBdr><w:bottom w:val="single" w:sz="6" w:space="8" w:color="000000"/></w:pBdr>'
                           '<w:spacing w:after="200"/></w:pPr>{}{}</w:p>'.format(run("Keywords: ", bold=True),
                                                                             inline(self.resolve(s[10:].strip()))))
                prev = "heading"
                continue
            m = re.match(r"^(#{1,3})\s+(.*?)\s*(\{#sec:[\w-]+\})?$", s)
            if m:
                flush()
                out.append(self.heading(len(m.group(1)), m.group(2)))
                prev = "heading"
                continue
            if s.startswith("- "):
                flush()
                out.append('<w:p><w:pPr><w:pStyle w:val="ListParagraph"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr>'
                           '<w:spacing w:after="60"/><w:jc w:val="both"/></w:pPr>{}</w:p>'.format(inline(self.resolve(s[2:]))))
                prev = "list"
                continue
            if s.startswith("@eq:"):
                flush()
                name = re.sub(r"\s*\{#eq:[\w-]+\}", "", s[4:]).strip()
                eq_count[0] += 1
                out.append(self.equation(name, eq_count[0]))
                prev = "equation"
                continue
            if s.startswith("@table:"):
                flush()
                out.append(self.table(s.split(":", 1)[1].strip()))
                prev = "table"
                continue
            if s.startswith("@fig:"):
                flush()
                out.append(self.figure(s[5:]))
                prev = "figure"
                continue
            if s.startswith("@algorithm:"):
                flush()
                name = s.split(":", 1)[1].strip()
                out.append('<w:p><w:pPr><w:spacing w:before="120"/></w:pPr></w:p>')
                out.append(open(os.path.join(HERE, "equations", name + ".xml"), encoding="utf-8").read())
                out.append('<w:p><w:pPr><w:spacing w:after="120"/></w:pPr></w:p>')
                prev = "table"
                continue
            if s.startswith("@references"):
                flush()
                out.append(self.reference_list())
                continue
            buffer.append(s)
        flush()
        if self.missing and not self.draft:
            raise SystemExit("values missing from values.json: " + ", ".join(sorted(self.missing)))
        if self.missing:
            print("draft: {} values missing".format(len(self.missing)))
        return "".join(out)


def png_size(path):
    with open(path, "rb") as handle:
        head = handle.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit("not a PNG: " + path)
    return struct.unpack(">II", head[16:24])


def write_docx(builder, out_path, title, authors):
    body = builder.body()
    template = os.path.join(HERE, "template")
    head = open(os.path.join(template, "document_head.xml"), encoding="utf-8").read()
    sect = open(os.path.join(template, "document_sectpr.xml"), encoding="utf-8").read()
    document = head + body + sect + "</w:body></w:document>"

    rels = open(os.path.join(template, "word", "_rels", "document.xml.rels"), encoding="utf-8").read()
    rels = re.sub(r'<Relationship Id="rId\d+" Type="[^"]*/image" Target="media/[^"]*"/>', "", rels)
    extra = "".join('<Relationship Id="{}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                    'Target="media/image{}.png"/>'.format(rid, i + 1) for i, (rid, _) in enumerate(builder.images))
    rels = rels.replace("</Relationships>", extra + "</Relationships>")

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            '<dc:title>{}</dc:title><dc:creator>{}</dc:creator><cp:lastModifiedBy>{}</cp:lastModifiedBy><cp:revision>1</cp:revision>'
            '<dcterms:created xsi:type="dcterms:W3CDTF">{}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{}</dcterms:modified>'
            '</cp:coreProperties>').format(escape(title), escape(authors), escape(authors.split(",")[0]), now, now)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(template, "[Content_Types].xml"), "[Content_Types].xml")
        z.write(os.path.join(template, "_rels", ".rels"), "_rels/.rels")
        z.writestr("docProps/core.xml", core)
        z.write(os.path.join(template, "docProps", "app.xml"), "docProps/app.xml")
        z.writestr("word/document.xml", document)
        z.writestr("word/_rels/document.xml.rels", rels)
        for name in ("styles.xml", "settings.xml", "fontTable.xml", "webSettings.xml", "numbering.xml",
                     "footer1.xml", "footnotes.xml", "endnotes.xml"):
            z.write(os.path.join(template, "word", name), "word/" + name)
        z.write(os.path.join(template, "word", "theme", "theme1.xml"), "word/theme/theme1.xml")
        for i, (_, path) in enumerate(builder.images):
            z.write(path, "word/media/image{}.png".format(i + 1))
    os.replace(tmp, out_path)


def export_pdf(docx_path, pdf_path):
    script = os.path.join(HERE, "docx_to_pdf.ps1")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script,
                    "-InputDocx", os.path.abspath(docx_path), "-OutputPdf", os.path.abspath(pdf_path)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", default=os.path.join(HERE, "content.md"))
    parser.add_argument("--generated", default=os.path.join(HERE, "generated"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--draft", action="store_true", help="allow missing values, tables and figures")
    args = parser.parse_args()

    tables = json.load(open(os.path.join(HERE, "tables_static.json"), encoding="utf-8"))
    gen_tables = os.path.join(args.generated, "tables_generated.json")
    if os.path.exists(gen_tables):
        tables.update(json.load(open(gen_tables, encoding="utf-8")))
    values_path = os.path.join(args.generated, "values.json")
    values = json.load(open(values_path, encoding="utf-8")) if os.path.exists(values_path) else {}
    references = json.load(open(os.path.join(HERE, "references.json"), encoding="utf-8"))

    builder = Builder(args.content, tables, values, references, args.generated, draft=args.draft)
    text = open(args.content, encoding="utf-8").read()
    title = re.search(r"^@title:\s*(.*)$", text, flags=re.M).group(1)
    authors = re.search(r"^@authors:\s*(.*)$", text, flags=re.M).group(1)
    write_docx(builder, args.out, title, authors)
    print("wrote {} ({} references cited, {} figures)".format(args.out, len(builder.cite_order), len(builder.images)))
    unused = [k for k in references if k not in builder.cite_order]
    if unused:
        print("references not cited (left out of the list): " + ", ".join(unused))
    if args.pdf:
        export_pdf(args.out, os.path.splitext(args.out)[0] + ".pdf")
        print("wrote", os.path.splitext(args.out)[0] + ".pdf")


if __name__ == "__main__":
    main()
