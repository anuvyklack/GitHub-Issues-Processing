# GitHub-Issues-Processing
This script uses the approach "bag with words". Each issue represented by a vector in vector space, where each unique word is an additional dimension in this vector space.
The tf-idf is used to calculate the weight of each term. The degree of similarity is the cosine of an angle between these vectors. This means, that the more two different issues consist of the same words, the higher their degree of similarity is.

Instruction:
* First run gitHub-collect-issues.py script. Use your GitHub login/password, or otherwise GitHub API will block you.
* Then run duplicates.py script.
