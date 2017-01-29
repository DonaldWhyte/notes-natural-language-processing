## Natural Language Processing Study Notes

TODO: intro to nlp

The notes in this repo explain TODO.

### Get the Source

```
git clone https://github.com/DonaldWhyte/notes-natural-language-processing.git
```

### Building the Docs

The notes are written in LaTeX. To build the notes, first install LaTeX and
pdflatex. [**This document**](https://en.wikibooks.org/wiki/LaTeX/Installation)
explains how to install LaTeX on Windows, Mac OS X and various Linux
distributions.

Afterwards, navigate into the cloned and execute the build script:

```
cd notes-natural-language-processing
make docs
```

This will build three PDF documents in the current working directory:

* `natural_language_processing.pdf` -- main document that contains almost all the notes
* `text-classification/nlp-cw1.pdf` -- document showing performance of different text classifiers and features to detect movie reiew sentiment
* `n-grams/nlp-cw2.pdf` -- document 

### Executing the Test Programs

See the READMEs of the individual coursework directories (`text-classification`
and `n-grams`) for instructions on how to run the test programs. These programs
generated the data that accompanied the written coursework documents.

### Acknowledgements

Credit to University of Leeds and Katja Markey for doing a great job
teaching me the fundamentals of natural language processing.
