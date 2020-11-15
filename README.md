## Instruction to run:
python main.py \<input file name\>

example : python main.py Ouput_From_Recognizer_Pos/test

This will store the ouput in "ouput.txt" file and also print in the standard output

## Format of input file:
each line in the file contains 6 tab separated values : 

			1. tags [tags extracted by the recongnizer, if not available, left blank]
			2. tweet
			3. creation date
			4. event date
			5. POS tags 

sample input file can be found in "Ouput_From_Recognizer_Pos" directory

## Details:

This project contains the Normalizer described in the the following paper:

[1] Jeniya Tabassum, Alan Ritter, Wei Xu. "TweeTIME : A Minimally Supervised Method for Recognizing and Normalizing Time Expressions in Twitter". EMNLP 2016.

The POS tags are generated with [twitter_nlp](https://github.com/aritter/twitter_nlp).

This project is maintained by [Jeniya Tabassum](https://sites.google.com/site/jeniyatabassum/). Feel free to contact tabassum.13@osu.edu for any relevant issue. This repo will continue to be updated.
