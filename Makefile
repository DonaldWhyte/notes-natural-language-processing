DOC_NAME = natural_language_processing

default: all

all: docs programs

clean: clean-docs clean-programs

programs:
	$(MAKE) -C text-classification
	$(MAKE) -C n-grams

clean-programs:
	$(MAKE) -C text-classification clean
	$(MAKE) -C n-grams clean

# Build document twice (first time to build TOC, second time to use it).
# Skip the first build if the toc index files have already been generated.
docs:
	@if [ ! -f $(DOC_NAME).toc ] ; \
	then \
	    pdflatex $(DOC_NAME).tex ; \
	fi;
	@pdflatex $(DOC_NAME).tex

clean-docs:
	@rm -f *.pdf *.aux *.lof *.log *.lot *.fls *.out *.toc *.fmt *.fot *.cb *.cb2

.PHONY: clean clean-programs clean-docs