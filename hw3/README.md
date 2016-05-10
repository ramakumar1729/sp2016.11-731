There are three Python programs here (`-h` for usage):

 - `./decode` a simple non-reordering (monotone) phrase-based decoder
 - `./decode_baseline`: our implementation (more details at the end)
 - `./grade` computes the model score of your output

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./decode_baseline | ./grade


The `data/` directory contains the input set to be decoded and the models

 - `data/input` is the input text

 - `data/lm` is the ARPA-format 3-gram language model

 - `data/tm` is the phrase translation model

I tried the idea that was hinted at which was to re-order phrases anywhere in the sentence. We tried different window sizes within which we re-order phrases but setting the window to 25 (roughly the max sentence length) worked best. We added a penalty term that penalized long-range re-orderings. Also, we re-ranked the k-best sentences in the final stack based on the actual grading script. K beyond ~ 10 gave us no improvements. Finally, we aggregated results from multiple models that we trained using the grade script again.

The first thing I tried was to be able to create hypotheses that swap adjoining phrases. This was able to beat the baseline on tuning the parameters such as size of beam.

Then I generalized it to geneate hypothesis with arbitrary reordering. This was done by inserting the last phrase at all possible positions before to generate so many hypothesis. A window limitation was set as to how far apart the insertions could be. A window size of around 25 was empirically found to be optimal. Long range reorderings were penalized. 
A re-ranking of the final k-best hypotheses was done based on the grading script. We also aggregated the outputs of multiple models to choose the best output.
