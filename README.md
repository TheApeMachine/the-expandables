# THE EXPANDABLES

A custom solution for keyword expansion using three popular systems combined
to try and improve relevancy.

## PROBLEM

All existing popular solutions (word2vec, wordnet, spacy) give us back too many
words that are not relevant to us, and after training on many corpora there
was little significant improvement.

## SOLUTION

Try to use a combination of word2vec, wordnet, and spacy to validate the output
that each one individual model returns.
