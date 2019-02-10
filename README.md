# Music Genre Classification using Recurrent Neural Networks

1. Clone the repo: `git clone https://github.com/Ornifex/ML.git`.

Do the following two steps to recompute the used features, otherwise go to step 4.

2. Download the [GTZAN data set](http://opihi.cs.uvic.ca/sound/genres.tar.gz) and extract into the repo folder.

3. Run feature extraction: `python features.py` to (re)generate features.csv.

4. Run the BLSTM *n* times: `python main.py n`.

## Random Hyperparameter Search
* For *n* trials simply run `python sweep.py n`.


