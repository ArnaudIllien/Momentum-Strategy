"""Rapport PDF : mêmes graphiques Plotly que le tableau de bord (export PNG via
kaleido + Chrome), mis en page avec fpdf2.

    pdf_bytes = build_pdf(title, subtitle, sections)

`sections` est une liste de blocs :
    ("h1", "Titre de section")
    ("text", "Paragraphe…")
    ("kv", {"Paramètre": "valeur", …})             -> tableau à 2 colonnes
    ("table", DataFrame déjà formaté en texte)
    ("fig", go.Figure, hauteur_px)
    ("pagebreak", None)
"""

from __future__ import annotations

import os
import warnings
from datetime import date

import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF

FONT_DIR = "C:/Windows/Fonts"
PAGE_W = 210          # A4 (mm)
MARGIN = 14
CONTENT_W = PAGE_W - 2 * MARGIN
INK, MUTED, RULE = (11, 11, 11), (82, 81, 78), (225, 224, 217)

# Remplacements si aucune police Unicode n'est disponible (polices PDF standard = latin-1)
_ASCII = {"—": "-", "–": "-", "→": "->", "≤": "<=", "≥": ">=", "×": "x", "÷": "/", "σ": "sigma",
          "Σ": "Somme", "’": "'", "“": '"', "”": '"', "…": "...", "·": "-", "−": "-", "ᵢ": "i",
          "€": "EUR", "\u202f": " ", "\u00a0": " "}


class _PDF(FPDF):
    def __init__(self, title: str):
        super().__init__(format="A4")
        self.doc_title = title
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(True, margin=16)
        reg, bold = os.path.join(FONT_DIR, "arial.ttf"), os.path.join(FONT_DIR, "arialbd.ttf")
        if os.path.exists(reg) and os.path.exists(bold):
            self.add_font("Body", "", reg)
            self.add_font("Body", "B", bold)
            self.font_name, self.unicode = "Body", True
        else:
            self.font_name, self.unicode = "Helvetica", False

    def txt(self, s) -> str:
        s = str(s)
        if self.unicode:
            return s
        for a, b in _ASCII.items():
            s = s.replace(a, b)
        return s.encode("latin-1", "replace").decode("latin-1")

    def footer(self):
        self.set_y(-12)
        self.set_font(self.font_name, "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, self.txt(f"{self.doc_title} · {date.today():%d/%m/%Y}"), align="L")
        self.cell(0, 6, f"{self.page_no()}/{{nb}}", align="R")


def _fig_png(fig: go.Figure, height_px: int) -> bytes:
    """PNG haute résolution, thème clair forcé (le PDF est imprimé sur fond blanc)."""
    f = go.Figure(fig)
    f.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                    font=dict(family="Arial", size=12, color="#0b0b0b"), height=height_px, width=1000)
    # Légende sous le graphique : jamais de chevauchement avec le titre, même sur 2 lignes
    f.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0),
                    margin=dict(l=10, r=10, t=50, b=40))
    f.update_xaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7")
    f.update_yaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", zerolinecolor="#c3c2b7")
    return f.to_image(format="png", scale=2)


def build_pdf(title: str, subtitle: str, sections: list[tuple]) -> bytes:
    import kaleido

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        server = False
        try:   # garde Chrome ouvert pendant tout le rapport : ~0,2 s par graphique au lieu de ~7 s
            kaleido.start_sync_server(silence_warnings=True)
            server = True
        except Exception:
            pass
        try:
            return _render(title, subtitle, sections)
        finally:
            if server:
                kaleido.stop_sync_server(silence_warnings=True)


def _render(title: str, subtitle: str, sections: list[tuple]) -> bytes:
    pdf = _PDF(title)
    pdf.alias_nb_pages()
    pdf.add_page()
    fn = pdf.font_name

    pdf.set_font(fn, "B", 20)
    pdf.set_text_color(*INK)
    pdf.multi_cell(CONTENT_W, 9, pdf.txt(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(fn, "", 10)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(CONTENT_W, 5, pdf.txt(subtitle), align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    for kind, payload, *extra in sections:
        if kind == "pagebreak":
            pdf.add_page()
        elif kind == "h1":
            if pdf.get_y() > 250:
                pdf.add_page()
            pdf.ln(3)
            pdf.set_font(fn, "B", 13)
            pdf.set_text_color(*INK)
            pdf.cell(CONTENT_W, 8, pdf.txt(payload), new_x="LMARGIN", new_y="NEXT")
            pdf.set_draw_color(*RULE)
            pdf.line(MARGIN, pdf.get_y(), PAGE_W - MARGIN, pdf.get_y())
            pdf.ln(2)
        elif kind == "text":
            pdf.set_font(fn, "", 9.5)
            pdf.set_text_color(*INK)
            pdf.multi_cell(CONTENT_W, 4.8, pdf.txt(payload), align="L", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif kind == "kv":
            _table(pdf, pd.DataFrame({"": list(payload.values())}, index=list(payload.keys())), first_w=0.45)
        elif kind == "table":
            _table(pdf, payload, first_w=extra[0] if extra else None)
        elif kind == "fig":
            h_px = extra[0] if extra else 380
            h_mm = CONTENT_W * h_px / 1000
            if pdf.get_y() + h_mm > 280:
                pdf.add_page()
            from io import BytesIO
            pdf.image(BytesIO(_fig_png(payload, h_px)), x=MARGIN, w=CONTENT_W)
            pdf.ln(2)
    return bytes(pdf.output())


def _table(pdf: _PDF, df: pd.DataFrame, first_w: float | None = None):
    """Tableau simple : index en 1re colonne, valeurs alignées à droite."""
    fn = pdf.font_name
    n = len(df.columns) + 1
    w0 = CONTENT_W * (first_w if first_w else min(0.34, 1.6 / n))
    wi = (CONTENT_W - w0) / max(n - 1, 1)
    row_h = 5.2
    show_header = any(str(c) for c in df.columns)
    if pdf.get_y() + row_h * (len(df) + 1) > 280:
        pdf.add_page()
    pdf.set_draw_color(*RULE)
    if show_header:   # en-têtes sur plusieurs lignes si nécessaire (pas de chevauchement)
        pdf.set_font(fn, "B", 7.5)
        pdf.set_text_color(*MUTED)
        n_lines = max(len(pdf.multi_cell(wi - 1, 4, pdf.txt(c), dry_run=True, output="LINES"))
                      for c in df.columns)
        h = max(row_h, 4 * n_lines + 1)
        y0, x = pdf.get_y(), MARGIN + w0
        for c in df.columns:
            lines = len(pdf.multi_cell(wi - 1, 4, pdf.txt(c), dry_run=True, output="LINES"))
            pdf.set_xy(x, y0 + h - 4 * lines - 1)
            pdf.multi_cell(wi - 1, 4, pdf.txt(c), align="R")
            x += wi
        pdf.set_y(y0 + h)
        pdf.line(MARGIN, pdf.get_y(), PAGE_W - MARGIN, pdf.get_y())
    pdf.set_font(fn, "", 8.5)
    pdf.set_text_color(*INK)
    for idx, row in df.iterrows():
        if pdf.get_y() + row_h > 282:
            pdf.add_page()
        pdf.cell(w0, row_h, pdf.txt(idx), border="B")
        for v in row.values:
            pdf.cell(wi, row_h, pdf.txt(v), border="B", align="R")
        pdf.ln(row_h)
    pdf.ln(2)
