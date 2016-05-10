There are two Python programs here:

 - `python bleu.py your-output.txt ref.txt` to compute the BLEU score of your output against the reference translation.
 - `python rnnlm.py ref.txt` trains an LSTM language model, just for your reference if you want to use pyCNN to perform this assignment.

The `data/` directory contains the files needed to develop the MT system:

 - `data/train.*` the source and target training files.

 - `data/dev.*` the source and target development files.

 - `data/test.src` the source side of the blind test set.


The model that worked best was a seq-seq neural network model with attention, using a bidirectional LSTM for encoding the source sequence and a LSTM for decoding target sequence. 
I tried out several ways to provide both context and encoder representation at each decoding step. 

Note: For this homework, I reused code that we developed as a part of the course project on 'Incremental Machine Translation'. 
