# Keyword extraction using TextRank
The aim of this project is to measure similarity between companies using their descriptions. Several different hand crafted methods have been used to create a similarity score (usually based on number of keywords in common, sometimes with some weight on the keywords). An example of the hand crafted methodis included in in the "new\_similarities.py" file.

Another approach was to use a vectorization method on the company descriptions, which could be used to calculate similarities simply by using the dot product (vectors are normalized). A few vectorization methods were attempted, the two most successful were simple LSA (TF-IDF + SVD), as well as a custom method based on the TextRank algorithm.

### Requirements

Python (3), with the packages specified in requirements.txt.

## TextRank

The vectorization based on TextRank is slighly involved, and therefore needs some explanation. First, a Word2Vec model is trained on a large corpus in order to create a word vectorization model. Then, the TextRank algorithm (a small modification from PageRank) is applied to each company description in order to extract top 10 most important words in each description. These vectors are averaged (can be weighted or unweighted) to produce the single vector for each description.

Similarity is then computed as a dot product, like after LSA.
