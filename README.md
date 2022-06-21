# The Noisy Channel and a Probabilistic Spell Checker

Without ML libraries in this assignment I builded a spell checker that handles both non-word and real-word errors given in a sentential context. 
In order to do that I had to learn a language model and use error matrices, combining it all to a context sensitive noisy channel model.

This assignment gave me hands-on experience with probabilistic language models by implementing the entire algorithmic pipeline.


Specifics:
- I used the noisy channel model to correct the errors, that is - a correction of a misspelled word depends on both the most probable correction on the error type-character level and words prior; A correction of a word in a sentence depends on the error type-character level and on the language model -- the correction should maximize the likelihood of getting the full corrected sentence. 
- I used the language model when correcting words in a sentential context. 
- I assumed a word has at most two errors (that is, ‘character’ will be considered as a correction for ‘karacter’ [sub+del], while it will not be considered for ‘karakter’).
- I assumed a sentence has at most one erroneous word.  
- I wanted to give the normalization of text function a twist, so I used the tragedy "Hamlet" that written by "Shakespeare" as my database.
