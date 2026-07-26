.PHONY: all clean distclean

all: paper.pdf

paper.pdf: paper.tex references.bib
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error paper.tex

clean:
	latexmk -c paper.tex

distclean:
	latexmk -C paper.tex
