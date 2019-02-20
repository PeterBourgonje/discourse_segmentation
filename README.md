# discourse_segmentation

A set of Python scripts to segment text into Elementary Discourse Units.

The scripts are designed to work with the (conll-format) data from the 2019 DISRPT shared task (https://github.com/disrpt/sharedtask2019).
The randomforest and lstm approaches perform the same task, but score different on the corpora from the shared task.


``baseline.py`` runs the baseline system, assuming a segment boundary at the start of every sentence, and if the ``-c`` option is present, also after every comma. Specify the path to the (conll-format) test file and the labels are predicted and written to a file suffixed .baseline.predicted next to the original test file.

``randomforest.py`` trains and runs a randomforest classifier (predicts on the test set). Specify the paths to the train, dev and test file (train and dev are thrown together for training) with the ``--train``, ``--dev`` and ``--test`` options and the labels are predicted and written to a file suffixed .randomforest.predicted next to the original test file.

``lstm.py`` trains and runs an lstm (predicts on the test set). Specify the paths to the train, dev and test file (train and dev are thrown together for training) with the ``--train``, ``--dev`` and ``--test`` options and the labels are predicted and written to a file suffixed .lstm_extembs.predicted next to the original test file.

Because the labels for the PDTB-style corpora in the shared task differ (``Seg=(B|I)-Conn`` instead of ``BeginSeg=Yes`` for all others), ``post_edit_pdtb_output.py`` modifies the labels accordingly. This script thus is only needed for PDTB-style corpora.

## Installation & Configuration

Install all python requirements (``pip3 install -r requirements.txt``).

For ``randomforest.py``, for the languages supported by the Stanford LexicalizedParser (German, English, French, Spanish and Chinese), this parser is used to extract features based on the (constituency) tree. Download the parser here: https://nlp.stanford.edu/software/lex-parser.shtml#Download

Then specify the paths accordingly in the config.ini file (the stanfordParser, stanfordModels and path properties) and make sure java (8) is installed.
Note that obtaining the parse trees can take a while (up to several hours). For the non-proprietary corpora in the shared task, the parse trees for all sentences are stored in the picklejar folder and read from there, speeding up the process considerably.

For ``lstm.py``, you have a choice of either using external/pre-trained embeddings, or get the embeddings from the corpus itself (currently, external embeddings are used. Change the relevant lines in main to call ``intembs`` instead of ``extembs``). The external embeddings can be downloaded from here: https://fasttext.cc/docs/en/crawl-vectors.html (the .vec files, i.e. cc.de.300.vec for German). 

Then specify the paths according in the config.ini file (the embeddings section).
