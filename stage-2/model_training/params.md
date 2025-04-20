## A. Data Statistics
### Some titles have been filtered out during cleaning.
- **Total Number of Source Titles**: 1,079,527
- **Total Number of Tokenized Titles**: 825,389

## B. If A and B are different, what have you done for that?
### The difference between the original (A) and tokenized (B) counts is due to data filtering.
- I filtered out titles that did not meet specific criteria.
    - For example, those starting with "re:" or "fw:" and titles with invalid or blank content. 
- This filtering process ensures that only clean, valid titles are used for further processing.

## C. Parameters of Doc2Vec Embedding Model.
### Data statistics
- **Total Number of Training Documents**: 825,389
- **Output Vector Size**: 60 
- **Min Count**: 2 
- **Epochs**: 300 
- **Workers**: 4
### The evaluation shows:
- **First Self Similarity**: 84.13% 
- **Second Self Similarity**: 86.69%

## D. Parameters of Multi-Class Classification Model.
### Architecture and Settings:
- **Layer Arrangement**: 60 x 200 x 200 x 200 x 200 x 9
- **Activation Functions**:
    - Hidden Layers: ReLU
    - Output Layer: Softmax
- **Loss Function**: Categorical Cross Entropy
- **Backpropagation Algorithm**: Adam
- **Training Documents**: 660,311
- **Testing Documents**: 165,078
- **Epochs**: 200
- **Learning Rate**: 0.001
### Results:
- **Train Accuracy**: 84.17%
- **Test Accuracy**: 81.73%

## F. Share your experience of optimization, including at least 2 change/result pairs.
### Embedding model(d2v model)
1. Vector size
    - Change: 100 -> 150 -> 50 -> 40 -> 60
    - Result: Changing the vector size did not have any significant effect on the first or secondary similarity.
2. Window
    - Change: 5 -> 2 -> 3
    - Result: Since the tokenized data often contains a limited number of words, a larger window resulted in worse accuracy. There was no significant difference between using a window size of 2 and 3, but using 2 reduced the computational load.
3. Epochs
    - Change: 20 -> 100 -> 200 -> 300
    - Result: Increasing the number of epochs significantly improved the accuracy when the epochs were initially too low(e.g., 20 or 100) .
4. Min count
    - Change: 3 -> 5 -> 2
    - Result: A larger min count significantly lowered the accuracy, possibly because the training data was not large enough to support the higher threshold.

### Classify model
1. Hidden layer depth
    - Change: 2 -> 3 -> 4 (Latest: 60 x 200x 200 x 200 x 200 x 9)
    - Result: Increasing the depth improved accuracy. A shallower model (e.g., 50 x 200 x 200 x 9) was not sufficient for handling the complexity of the data.
2. Hidden layer dimension
    - Change: 100 -> 150 -> different sizes each layer -> 200
    - Result: Increasing the hidden layer dimensions improved accuracy; smaller dimensions appeared insufficient for effectively modeling the data.
3. Epochs
    - Change: 100 -> 200
    - Result: Increasing the number of training epochs led to improved performance, likely because the larger model requires more epochs to converge. (Refer to the logs in "stage-2\classify_models" for more details.)