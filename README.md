Spam SMS Classifier
A machine learning-based SMS classifier that distinguishes between spam and non-spam (ham) messages. This project demonstrates various classification algorithms and text preprocessing techniques to build an effective spam detection system.

Features
Text Preprocessing: Includes tokenization, stopword removal, and text vectorization using TF-IDF.
Multiple Classifiers Tested:
Support Vector Classifier (SVC)
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Multinomial Naive Bayes (Best Performer)
Bernoulli Naive Bayes
Decision Tree Classifier
Logistic Regression
Random Forest Classifier (2nd Best Performer)

Performance Metrics: Evaluated using accuracy, precision.
Best Model Performance
Multinomial Naive Bayes achieved the highest accuracy and precision, making it the most suitable model for this dataset.
Random Forest Classifier came in second, providing strong performance in terms of both accuracy and precision.

Tools & Technologies
Programming Language: Python

Libraries Used:
NumPy: Numerical computations
Pandas: Data manipulation
Matplotlib: Data visualization
Scikit-learn: Machine learning algorithms
NLTK: Natural language processing

Dataset
The project uses a dataset of SMS messages labeled as spam or ham. Each message is preprocessed before being fed into the classifiers for training and testing.
