# AWS ML Speciality

1. Prerequisite
   1. Distribution
     A. Poisson distribution -A Poisson distribution is a probability distribution that is used to show how many times an event is likely to occur over a specified period. In other words, it is a count distribution.
     B. Binomial distributions are specifically used for binary outcomes
   2. Redshift
     A. Use Redshift Spectrum to query data directly in S3
   3. Term Frequency
      A. Tf-idf = comparing relevancy of a word within documents. If the same word exist in both documents, then no tf-idf=0
      B. Unigrams, bigrams(!!)
        1. Please call the number below. 
        2. Please do not call us.
        Dimension = 2,16

2. Data processing
   1. GLUE -
      1. Glue Crawlers can help build the Glue Data Catalog
      2. Utilize AWS Glue's FindMatches ML Transform to detect and remove duplicate records upon their arrival.
   3. EMR
     1. 1 Master node, 1 Core node, a lot of Task node (Spot)
     2. Utilize Amazon EMR File System (EMRFS) to establish a connection between MapReduce and S3, employing the `s3://` file prefix for data access.
     3. Useful for converting big big big data 1TB
   5. Datapipeline -> move data
   6. Quickinsight -> Fast, easy, cloud-powered business analytics service (Anomaly detection !!!, Forecasting, Auto-narrative, visualise) -> connect to different data
   7. S3 ->
       1. S3 = Data Lake (to load data = COPY)
       2. S3 ORC, Parquet saves a lot of money for query!!
EG 
Transform the data from JSON format to Apache Parquet format using an AWS Glue job. Configure AWS Glue crawlers to discover the schema and build the AWS Glue Data Catalog. Leverage Amazon Athena to create a table with a subset of columns. Set up Amazon QuickSight for visual analysis of the data and identify fraudulent transactions using QuickSight's built-in machine learning-powered anomaly detection

3. Kinesis
number_of_shards = max (incoming_write_bandwidth_in_KB/1000, outgoing_read_bandwidth_in_KB/2000)
. Kinesis Streams: low latency streaming ingest at scale
• Kinesis Analytics: perform real-time analytics on streams using SQL, 
• Kinesis Firehose: load streams into S3, Redshift, ElasticSearch & Splunk
• Kinesis Video Streams: meant for streaming video in real-time

4. Feature engineering
   1. Dimensionality reduction (PCA/ Kmeans)
      PCA
         Regular -> sparse,
         Randomised -> large number of observation and feature
   3. Imputing Missing Data Dropping - Mean/ Median (Median is better) (NaN, not very accurate) / KNN / Most advanced technique: MICE OR use supervised learning to estimate
   4. Solve bias: Oversampling (increase the minority cause minority to little, fraud, Synthetic Minority Over-sampling TEchnique (SMOTE), generate through KNN) vs Undersampling (majority has many redundant)
   5. Adjusting Threshold -> Flag Positive!! (Precision) Too many false positives 
   6. Outliers - Standard Deviation, random cut forest
   7. Normalisation - some values will be too big, you still want to know the proportion, just too big so it does not fix the outliner, converge faster
   8. Shuffling -> Shuffle first then split it to different dataset
   9. AWS Ground Truth =  manages humans who will label your data for training purposes
   10. Hyperparameter = Use logarithmic scales on your parameter ranges , Decrease the number of concurrent hyperparameter tuning jobs
   11. Quantile binning  -> t continuous blood pressure readings 
   12. One-hot encoding is a method of representing categorical data (handwritten text). Transforming data into some new representation 
   13.  Substitute variable- Add substitute variables for each missing feature – when the feature has a missing value for a sample, set the substitute variable to 1 for that feature, and when the feature has a valid value, set the variable to 0
  
   eg
   Drop a feature if variance is small, lots of missing value and low correlation to target
   Replicate only a subset of data -> Set the S3DataDistributionType field to ShardedByS3Key

5. Neural Network

   1. Feedforward, Convolutional (Image Classification), RNN (LSTM)
   2. Learning -> learning rate, batch size (batch size smaller, more accurate, find global, longer time) LEARNING RATE/Batch size does not affect the OVERFITTING THING, more like the optimal solution
   3. Activation function ->
         1. Multiple Classification = Softmax,
         2. RNN = Tanh,
         3. other things = ReLU, Leaky ReLU,  PReLU, Maxout, Swish
   4. Overfitting -> 
       1. Regularisation L1 and L2 Regularization one is w; one is w2
          1. L1 Regularization works by eliminating features that are not important.
          2.  L2 Regularization keeps all the features but simply assigns a very small weight to features that are not important) L2 more efficient but L1 can reduce dimension | regularisation value is added to the loss function
       2. Dropout (too many layers, less likely to overfit)/ Early Stopping if validation drops too much
       3. Use more training data, Use less features in the model
   5. Train, Validation (hyper tuning parameters), Testing
   6. A residual for an observation in the evaluation data is the difference between the true target and the predicted target.

6. Performance Evaluation
   1. Recall - True Positive / (TP + FN) (when false negatives is important) (sensitivity) (ALL real positives)
   2. Precision - True Positive / (TP + FP) (ALL tested positive) (Prioritise precision if you are more aware of real negatives not being marked as positive)
   3. True Negative Rate = TN/TN+FP (ALL Real Negative)
   4. False Positive Rate = FP / FP +TN (ALL Real Negative) ; Specificity = true negative rate = TN /TN+FP
   5. F1 Score = 2* Precision * Recall / (Prevision + Recall) harmonic mean of prevision and sensitivity; higher = better balance between recall and prevision
   6. Root mean squared error = > root mean squared error only cares about right/wrong answers
   7. ROC curve (Area under the Curve AUC)
         True positive rate vs false positive rate  0.5 = useless 1 = perfect true positive rate (recall) vs false positive rate -> compare classifiers!  (Good for binary)
   9. PR Curve - Precision /Recall Curve , higher area under curve Precision, up right the more the btter (imblanced)
   10. Hypertuning 
      - Choose logarithmic scaling when you are searching a range that spans several orders of magnitude
      - AWS offers hyperband automation!

