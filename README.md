# Hotel Cancellation Prediction
## Introduction
For this project, we analyze a comprehensive dataset which contains customers' information collected from several hotels. For this dataset, the dimension is 119390×32. Within the 32 features, 'is_canceld' is what we want to
predict. Our main contribution is pre-process the categorical features with word embedding and clustering, and evaluate the performance of different machine learning algorithms. 
The accuracy, recall, precision, and F1-score we dinally achieved all exceeded 99%. 

## Feature Engineering
The standout feature of our project is our use of word embeddings combined with clustering to manage categorical features with an extensive number of labels. For instance, in our dataset, the 'country' feature includes nearly 200 labels. Utilizing one-hot encoding for such features would significantly increase the dataset's dimensionality and lead to extreme sparsity. To address this issue, we implemented the following process:

Step 1: We vectorized the country names using OpenAI's word embedding model, which outputs vectors of dimension 1536.

Step 2: We applied non-linear dimensionality reduction techniques, such as Kernel PCA, to reduce the vector dimensions from 1536 to 2.

Step 3: We used clustering on these 2-dimensional vectors to reassign labels to the original countries. In our project, the clustering identified five distinct groups. Consequently, we relabeled the 'country' feature based on these results, reducing the feature's dimensionality to 5.







