SHELL = /bin/bash

MAIN    = thesis
LATEX   = pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX  = bibtex
MKINDEX = makeindex

PAGES   = $(wildcard Pages/*.tex)
SOURCES = $(MAIN).tex $(MAIN).bib cam-thesis.cls $(PAGES) \
          $(wildcard Pages/Figures/CollegeShields/*.pdf)

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(SOURCES)
	$(LATEX)   $(MAIN)
	$(BIBTEX)  $(MAIN)
	$(MKINDEX) $(MAIN)
	$(LATEX)   $(MAIN)
	$(LATEX)   $(MAIN)

clean:
	rm -f $(MAIN).pdf
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).bbl $(MAIN).blg
	rm -f $(MAIN).toc $(MAIN).out $(MAIN).lof $(MAIN).lot
	rm -f $(MAIN).idx $(MAIN).ilg $(MAIN).ind
	find Pages -name '*.aux' -delete 2>/dev/null; true
