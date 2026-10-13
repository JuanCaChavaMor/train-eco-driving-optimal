# Build the report:  make        (tables and figures from results/, then the PDF)
PY ?= python3

all: report.pdf

report.pdf: report.tex references.bib tables
	latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex

tables:
	$(PY) make_tables.py

figures:
	$(PY) make_figures.py

clean:
	latexmk -C

.PHONY: all tables figures clean
