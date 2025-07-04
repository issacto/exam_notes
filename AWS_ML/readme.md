# AWS ML Speciality

---

## 1. Prerequisites
A data warehouse can only store structured data whereas a data lake can store structured, semi-structured and unstructured data
How many shards would a Kinesis Data Streams application need if the average record size is 500KB and 2 records per second are being written into this application that has 7 consumers?
4!!!!
Models stored as  model.tar.gz

Linear Learner, and XGBoost  do not support incremental training.

Kinesis Data Streams PutRecord API uses name of the stream, a partition key and the data blob

Kinesis Data Firehose PutRecord API uses the name of the delivery stream and the data record
When a model underfits, it exhibits low accuracy on both the training and test data.

Use more training data helps overfitting!!!!!

When a model is underfitting, then adding more features!!!!

Transfer learning (use pertained model to train something else) vs Incremental Learning (Train the same model)

Word2Vec is only for word. Object2vec 

Copy the summary text fields and use them to fill in the missing full review text fields, and then run
through the test set. 
 

### A. Distribution

* **Poisson Distribution**: A probability distribution used to show how many times an event is likely to occur over a specified period. It's a count distribution.
* **Binomial Distribution**: Specifically used for binary outcomes.

### B. Redshift

* **Redshift Spectrum**: Use to query data directly in S3.

### C. Term Frequency

* **Tf-idf**: Compares the relevancy of a word within documents. If the same word exists in both documents, then tf-idf = 0.
* **Unigrams, Bigrams**:
    * Example: "Please call the number below."
    * Example: "Please do not call us."
    * Dimension = 2,16

### D. NLP (Sentence Analysis)

* **Tokenization**: Breaking text into individual words or units.
* **Lowercasing**: Converting all text to lowercase.
* **Stop Word Removal**: Removing common words (e.g., "the," "is") that don't add much meaning.

---

## 2. Data Processing

### A. AWS Glue

* **Glue Crawlers**: Help build the Glue Data Catalog.
* **AWS Glue's FindMatches ML Transform**: Detect and remove duplicate records upon their arrival.

### B. Amazon EMR

* **Node Configuration**: Typically 1 Master node, 1 Core node, and many Task nodes (often using Spot Instances).
* **Amazon EMR File System (EMRFS)**: Establishes a connection between MapReduce and S3, employing the `s3://` file prefix for data access. = transfer different files
* **Use Case**: Useful for converting very large data (e.g., 1TB).

### B1. Amazon Batch
* ** Serverless docker
* Step function better than batch when orchestrating a sequence of dependent tasks


### C. AWS Data Pipeline

* **Purpose**: Used to move data between different AWS services.

### D. Amazon QuickSight

* **Functionality**: Fast, easy, cloud-powered business analytics service.
* **Features**: Includes **Anomaly Detection**, **Forecasting**, **Auto-narrative**, and **Visualization**. Connects to various data sources.

### E. Amazon S3

* **Data Lake**: S3 acts as a data lake (use `COPY` command to load data).
* **Cost Savings**: Using **ORC** and **Parquet** formats in S3 saves significant money for queries.
* **Pipe Mode**: Streams data directly from S3 to the algorithm, reducing the need to use and store data on EBS volumes, thereby lowering costs. Ideal for large datasets.

### F. CSV Data Format

* **Common Format**: Usually, the first column is the label.
* **Format Specification**: `text/csv;label_size=0`
* Other format:
*    RecordIO format is Image
*    Parquet columnar text data,


### G. Example Data Transformation Flow

Transform data from JSON to Apache Parquet format using an AWS Glue job. Configure AWS Glue crawlers to discover the schema and build the AWS Glue Data Catalog. Leverage Amazon Athena to create a table with a subset of columns. Set up Amazon QuickSight for visual analysis of the data and identify fraudulent transactions using QuickSight's built-in machine learning-powered anomaly detection.

---

## 3. Kinesis

* **Number of Shards Calculation**: `number_of_shards = max(incoming_write_bandwidth_in_KB / 1000, outgoing_read_bandwidth_in_KB / 2000)`
* **Kinesis Streams**: Low-latency streaming ingest at scale.
* **Kinesis Analytics**: Perform real-time analytics on streams using SQL. -> running detection
* **Kinesis Firehose**: Load streams into S3, Redshift, Elasticsearch & Splunk.

 - Ingest the data using Kinesis Firehose that further transforms the data into Parquet format while writing to S3. Use an AWS Glue Crawler to read this data via an Athena table for ad-hoc analysis,
 - cannot convert data in RecorIO-Protobuf

* **Kinesis Video Streams**: Meant for streaming video in real-time.

---

## 4. Feature Engineering

### A. Dimensionality Reduction

* **PCA (Principal Component Analysis)**:
    * **Regular PCA**: Transforms regular data into sparse data.
    * **Randomized PCA**: Suitable for a large number of observations and features.
* **K-Means**: Used for clustering, which can sometimes aid in dimensionality reduction by grouping similar data points. Elbow Method is a popular technique for determining k in k-Means clustering

### B. Imputing Missing Data

* **Dropping**: Removing records with missing values (can lead to data loss).
* **Mean/Median Imputation**:
    * **Mean**: Filling missing values with the mean of the column.
    * **Median**: Filling missing values with the median of the column (often preferred for skewed data).
* **KNN Imputation**: Using K-Nearest Neighbors to estimate missing values.
* **Most Advanced Techniques**: **MICE (Multiple Imputation by Chained Equations)** or using supervised learning to estimate missing values.
* Substitute Variables
* **Method**: Add substitute variables for each missing feature. When a feature has a missing value for a sample, set the substitute variable to 1 for that feature, and when it has a valid value, set the variable to 0.


### C. Solving Bias

* **Oversampling**: Increasing the number of minority class samples when the minority class is too small (e.g., fraud detection).
    * **SMOTE (Synthetic Minority Over-sampling TEchnique)**: Generates synthetic minority samples through K-NN.
* **Undersampling**: Reducing the number of majority class samples when the majority class has many redundant instances.

### D. Adjusting Threshold

* **Purpose**: To flag positive cases, especially when **Precision** is a priority (e.g., avoiding too many false positives).

### E. Outliers

* **Detection Methods**:
    * **Standard Deviation**: Identifying data points far from the mean.
    * **Random Cut Forest**: An unsupervised anomaly detection algorithm.

### F. Normalization

* **Purpose**: Scales values to a standard range (e.g., 0 to 1 or -1 to 1) to prevent features with large values from dominating the learning process. It helps models converge faster but doesn't fix outliers.

### G. Shuffling

* **Process**: Shuffle data first, then split it into different datasets (training, validation, testing) to ensure random distribution.
* EXAMPLE
* Use 10,000 hours of clean speech data for training the model. Divide 100 hours of noisy data into validation and test sets. Optimize the model to improve validation performance and perform the final test using the test set

### H. AWS Ground Truth

* **Function**: Manages human labelers for training data.

### I. Hyperparameter Tuning

* **Techniques**:
    * Use **logarithmic scales** on parameter ranges when searching across several orders of magnitude.
    * Decrease the number of concurrent hyperparameter tuning jobs.

### J. Quantile Binning

* **Use Case**: For continuous data like blood pressure readings, dividing data into bins based on quantiles.

### K. One-Hot Encoding

* **Purpose**: A method of representing categorical data (e.g., handwritten text) by transforming it into a new numerical representation where each category becomes a binary (0 or 1) feature.


### M. Example Feature Engineering Scenarios

* Drop a feature if its variance is small, has many missing values, and low correlation to the target.
* To replicate only a subset of data, set the `S3DataDistributionType` field to `ShardedByS3Key`.
* Shuffle the training data and create a 5GB slice of this shuffled data. Build your model on the Jupyter Notebook using this slice of training data. Once the evaluation metric looks good, create a training job on SageMaker infrastructure with appropriate instance types and counts to handle the entire training data.

---

## 6. Neural Networks

### A. Types of Neural Networks

* **Feedforward Neural Networks**: Basic neural networks where information moves in only one direction.
* **Convolutional Neural Networks (CNN)**: Primarily used for image classification.
* **Recurrent Neural Networks (RNN)**: Used for sequential data, with **LSTM (Long Short-Term Memory)** being a popular variant.

### B. Learning Process

* **Learning Rate**: Determines the step size at each iteration while moving toward a minimum of the loss function.
* **Batch Size**: The number of training examples used in one iteration. Smaller batch sizes can lead to more accurate results and help find a global optimum but take longer.
    * *Note*: Learning rate and batch size affect the optimization process and convergence, but not directly overfitting.

### C. Activation Functions

* **Softmax**: Used for multiple classification problems.
* **Tanh (Hyperbolic Tangent)**: Often used in RNNs.
* **ReLU (Rectified Linear Unit)**, **Leaky ReLU**, **PReLU**, **Maxout**, **Swish**: Commonly used in various neural network architectures for their non-linear properties. -> useful for many layers

### D. Overfitting

* **Mitigation Techniques**:
    * **Regularization (L1 and L2)**: Added to the loss function to penalize large weights.
        * **L1 Regularization (Lasso)**: Works by eliminating unimportant features (can lead to sparse models and feature selection).
        * **L2 Regularization (Ridge)**: Keeps all features but assigns very small weights to unimportant ones (more efficient than L1 but doesn't reduce dimensions).
    * **Dropout**: Randomly sets a fraction of neurons to zero during training, preventing co-adaptation of neurons (less likely to overfit with more layers).
    * **Early Stopping**: Halting training when validation performance starts to drop.
    * **More Training Data**: Increasing the amount of data available for training.
    * **Less Features in the Model**: Reducing the complexity of the model by using fewer input features.
    * **Use more features in the training data,Use more layers in the network -> Make overfitting worse


### E. Model Splitting

* **Training Set**: Used to train the model.
* **Validation Set**: Used for hyperparameter tuning and early stopping.
* **Testing Set**: Used for final evaluation of the model's performance on unseen data.

### F. Residual

* **Definition**: For an observation in the evaluation data, the residual is the difference between the true target and the predicted target.

### G. Ensemble Methods

* **Bagging (Bootstrap Aggregating)**: Creates multiple models by training on different subsets of the training data. Averages their predictions to reduce variance and avoid overfitting.
* **Boosting**: Sequentially builds models where each new model corrects the errors of the previous ones. Often yields better accuracy.

### H. Example Training Scenario

* When training a deep learning model, if you increase the batch size, you should also **increase the learning rate**.

---

## 8. Performance Evaluation

### A. Classification Metrics

* **Recall (Sensitivity)**: `True Positive / (True Positive + False Negative)`. Important when false negatives are critical (e.g., detecting all real positives).
* **Precision**: `True Positive / (True Positive + False Positive)`. Prioritize precision if you are more concerned about real negatives not being marked as positive (i.e., minimizing false positives).
* **True Negative Rate (Specificity)**: `True Negative / (True Negative + False Positive)`. Measures the proportion of actual negatives that are correctly identified.
* **False Positive Rate**: `False Positive / (False Positive + True Negative)`. The proportion of actual negatives that are incorrectly identified as positive.
* **F1 Score**: `2 * (Precision * Recall) / (Precision + Recall)`. The harmonic mean of precision and recall. A higher F1 score indicates a better balance between recall and precision.

### B. Regression Metrics

* **Root Mean Squared Error (RMSE)**: A measure of the magnitudes of the errors between predicted and actual values. It primarily cares about the magnitude of right/wrong answers.

### C. Curve-Based Evaluation

* **ROC Curve (Receiver Operating Characteristic Curve)**: Plots the True Positive Rate vs. the False Positive Rate.
    * **Area Under the Curve (AUC)**: A common metric for ROC curves. 0.5 indicates a useless classifier, while 1 indicates a perfect classifier. Good for binary classification and comparing classifiers.
* **PR Curve (Precision-Recall Curve)**: Plots Precision vs. Recall. A higher area under the curve indicates better performance, especially for imbalanced datasets. The curve moves towards the upper right for better performance.

### D. Hyperparameter Tuning

* **Scaling**: Choose **logarithmic scaling** when searching a range that spans several orders of magnitude.
* **Automation**: AWS offers **Hyperband automation** for hyperparameter tuning.

### E. Example Threshold Adjustment

* Decreasing the class probability threshold makes the model more sensitive and, therefore, marks more cases as the positive class (e.g., fraud).

---

## 9. SageMaker Algorithms

### A. Supervised Learning Algorithms

* **Linear Learner**: For linear regression and classification.
* **XGBoost (eXtreme Gradient Boosting)**: A powerful tree-based ensemble algorithm.
    * **Data Format**: CSV must **not** have a column header record. The target variable must be the first column.
    * **Hyperparameters**:
        * **`subsample`**: Smaller values reduce overfitting.
        * **`eta`**: Learning rate, reduces step size to prevent overfitting.
        * **`gamma`**: Minimum loss reduction required to make a further partition on a leaf node; larger values lead to less overfitting.
        * **`alpha` (L1 Regularization)**: Larger values lead to less overfitting.
        * **`lambda` (L2 Regularization)**: Larger values lead to less overfitting.
        * **`eval_metric`**: Use AUC if false positives are a primary concern.
        * **`scale_pos_weight`**: Adjusts the balance of positive and negative classes, helpful for unbalanced datasets.
        * **`max_depth`**: Maximum depth of a tree; too high will lead to overfitting.
* **Seq2Seq**: Used for sequence-to-sequence tasks like machine translation, speech-to-text, and summarization.
* **DeepAR**: An RNN-based algorithm for time series forecasting.
* **BlazingText**: For text classification and **Word2Vec**.
    * **Word2Vec**: Generates embeddings for individual words, not entire documents. Can be used for classification of single words. Example format: `__label__4 linux ready for prime time , intel says.` Need space for special character
* **Object2Vec**: Similar to Word2Vec but for arbitrary objects, used to find similarities.
* **Object Detection**: Identifies and localizes objects within images or videos.
* **Image Classification**: Categorizes images into predefined classes.
* **Semantic Segmentation**: Pixel-level classification (e.g., medical imaging, self-driving cars).
* **KNN (K-Nearest Neighbors)**: A non-parametric classification algorithm.
* **Factorization Machines**: Good for recommendation systems with sparse data (requires `float32`).

### B. Unsupervised Learning Algorithms

* **Random Cut Forest**: Unsupervised deep learning algorithm for anomaly detection and outlier removal (e.g., fraud detection) ( Box plot, Histogram, Scatter plot)
* **Neural Topic Model (NTM)**: Unsupervised algorithm to classify or summarize documents.
* **LDA (Latent Dirichlet Allocation)**: A "bag-of-words" model (word order doesn't matter) for topic modeling not summary!!. Observations are documents, features are vocabulary, a feature is a word, and categories are topics.
* **K-Means**: A clustering algorithm.
    * **Elbow Method**: Apply by analyzing a plot of the total within-cluster sum of squares (WSS) against the number of clusters (`k`) to find the optimal `k`.
    * **Hyperparameters**: `feature_dim`, `k`.
* **PCA (Principal Component Analysis)**: Unsupervised dimensionality reduction.
* **IP Insights**: Unsupervised algorithm for detecting abnormal or anonymous IP addresses.

### C. Other SageMaker Tools and Integrations

* **`sagemaker-spark` library**: Integrates SageMaker with Apache Spark.
* **IAM Identity-Based Policies**: Specify allowed or denied actions, resources, and conditions for Amazon SageMaker.
* **Resource Tag Authorization**: Amazon SageMaker supports authorization based on resource tags.

---

## 10. High-Level Amazon AI Services

### A. Language Services

* **Amazon Comprehend**: NLP service for entity recognition, sentiment analysis, and topic comprehension. Can process non-English content without prior translation.
* **Amazon Translate**: Provides neural machine translation.
* **Amazon Transcribe**: Converts speech to text.
* **Amazon Polly**: Converts text to speech. Supports Speech Synthesis Markup Language (SSML) for control over pronunciation and whispering.
* **Amazon Lex**: Powers conversational interfaces (like Alexa), can take speech input.

### B. Vision Services

* **Amazon Rekognition**: Computer vision service for detecting celebrities, objects, scenes, and activities. Includes **Amazon Rekognition Video** for video analysis.

### C. Forecasting and Recommendation Services

* **Amazon Forecast**: An "AutoML" service that automatically chooses the best model for time series forecasting.
* **Amazon Personalize**: A real-time personalization and recommendation service.

---

## 11. SageMaker Pre and Post Deployment

---

### A. Training

* **Training Environments**: SageMaker Notebooks, Console, API.
    * **Jupyter Notebook**:
        * Save files in `/home/ec2-user/SageMaker` to prevent loss on instance restarts.
        * Specify the output path on an Amazon S3 bucket where the trained model will persist.

### B. Inference

* **Edge Computing**: **AWS SageMaker Neo** (compiler) + **AWS IoT Greengrass** (runtime) enables model deployment to edge devices.
* **Batch Inference**: **SageMaker Batch Transform** is used for non-real-time, large-scale batch predictions.
* **Distributed GPU Training**: Uses frameworks like **Horovod** on **Accelerated Computing / AI GPU instances** (e.g., g3, g4, p2, p3).
* **SageMaker Autopilot**: Automates model building, including **Hyperparameter Optimization (HPO)** and **Ensembling** (combining up to 10 models).
* datasets > 100MB = Autopilot ;< = ensembling.
* **Inference Pipelines**: Used for real-time or batch predictions, allowing for 2-15 containers in a sequence.
    * Can use pre-built TensorFlow Docker images provided by SageMaker to train and host models.
    * Can use spark
    * Can use for real time or batch
* **Connection and Security**:
    * Virtual Private Cloud (VPC) = connecting with other services
    * Direct Connect = connection on premise,  VPC interface endpoint connects your VPC 
    * Use **AWS Key Management Service (AWS KMS)** to manage encryption keys for data at rest and **TLS** for data in transit.
    * Enable **network isolation** for training jobs and models.
    * Sagemaker to S3 Access is limited to S3 buckets with "sagemaker" in the name, unless S3FullAccess is added.
* **Elastic Inference**: Provides GPU-powered acceleration to EC2 and SageMaker instances at a fraction of the cost of full GPU instances.

Must respond to /invocations and /ping requests on port 80.
Must accept all socket connection requests within 250 ms.

---

### C. Monitoring

* **Amazon CloudWatch**: For system and application-level metrics. SageMaker monitoring metrics are available at a 1-minute frequency. CloudWatch keeps statistics for 15 months, but the console limits searches to metrics updated in the last 2 weeks.
* **AWS CloudTrail**: Logs user activities and API calls made in Amazon SageMaker. Provides a record of actions taken by a user, role, or an AWS service. Keeps records for 90 days.




"Use Transfer Learning by removing the output layer of the image classification model, reinitialize the weights of the last hidden layer, and retrain the model."
