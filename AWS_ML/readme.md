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
   4. NLP (sentence analysis) - Tokenise, lowercase and remove stop words

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
       3. Using Pipe mode streams data directly from S3 to the algorithm, reducing the need to use and store data on EBS volumes, hence lowering costs associated with EBS. This mode is ideal for large datasets.
   8. CSV usually first column = label, text/csv;label_size=0
      
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
   Shuffle the training data and create a 5GB slice of this shuffled data. Build your model on the Jupyter Notebook using this slice of training data. Once the evaluation metric looks good, create a training job on SageMaker infrastructure with the appropriate instance types and instance counts to handle the entire training data


6. Neural Network

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
   7. Bagging and Boosting
      - Bagging = ensemble, avoids overfitting
      - Boosting = add weight to wrong ones , yields better accuracy

   eg
   When training a deep learning model, if you increase the batch size, you should also Increase the learning rate


8. Performance Evaluation
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
  
   eg
   Decreasing the class probability threshold makes the model more sensitive and, therefore, marks more cases as the positive class, which is fraud in this case.
   
9. Sagemaker Algorithms
  1. Linear learner
   2. XGboost (tree) (you can customize your own training scripts)
       1. XGboost = CSV must not have a column header record!!!.  Target variable must be the first column
       2. Hyperparameters
           1. Subsample - smaller,  less overfitting
           2. ETA - reduce step size, prevent overfitting
           3. Gamma - minimum loss reduction to create a partition , larger = less overfitting
           4. Alpha - L1 Regularisation, larger less overfitting
           5. Lambda -L2 , larger less overfitting
           6. Eval_metr-> if care about false positive then use AUC
           7. Scale_pos_weight (adjust balance of positive and negative , helpful for unbalanced class)
           8. Max_depth ( too high will overfit)
   3. Seq2Seq (translation/speech2Text/summarisation)
   4. DeepAR (time series) RNN
   5. BlazingText (text classification/ word2vec) - > Skip-gram model focuses on generating embeddings for individual words rather than entire tweets, classification Just a gsingle word!!, find classification. Word2vec only single WORD!!!! `__label__4 linux ready for prime time, intel says.`
   6. Object2Vec (word2vec but for things) find similarity
   7. Object Detection
   8. Image Classification
   9. Semantic Segmentation (medical/self driving) pixel-level  classification
   10. Random Cut Forest -> Unsupervised deep learning, Anomaly detection / outliner removal, unsupervised, fraud detection
   11. Neural Topic Model (Classify or summarize documents) (unsupervised)
   12. LDA ( Latent Dirichlet Allocation, topic) things other than words  (unsupervised)  LDA is a "bag-of-words" model, which means that the order of words does not matter Observations are referred to as documents. The feature set is referred to as vocabulary. A feature is referred to as a word. And the resulting categories are referred to as topics 
   13. KNN (Nearest-Neighbors) classification 
   14. KMeans (clustering) - Apply the "elbow method" by analyzing a plot of the total within-cluster sum of squares (WSS) against the number of clusters (k).
       1. Must Hyperparameter -> feature_dim,k 
   15. PCA (unsupervised dimensional reduction)
   16. Factorisation Machines (recommendation sparse) (float32)
   17. IP Insights unsupervised(abnormal/annonymous IP address)
   18.  sagemaker-spark library
   19.  With IAM identity-based policies, you can specify allowed or denied actions and resources as well as the conditions under which actions are allowed or denied for Amazon SageMaker
   20. Amazon SageMaker supports authorization based on resource tags

10. High level Amazon Service
   1. Amazon Comprehend -> NLP / Entities / Sentiment / Topic Comprehension ; Can take non English content! No need translation
   2. Amazon Translate
   3. Amazon Transcribe (Speech to Text)
   4. Amazon Polly (Text to Speech, Speak) (Synthesis Markup Language SSML to give control to pronunciation / whispering)
   5. Rekognition (Computer Vision) -> detect celebrities, Amazon Rekognition Video
   6. Forecast (“AutoML” chooses best model for your )
   7. Lex (Alexa) -> can also take speech
   8. Personalise  (recommendation)

11. Sagemaker Pre and Post Deployment
     1. Training
         1. SageMaker Notebook, Console, API
            Jupyter notebook
               Save as /home/ec2-user/SageMaker then notebook won’t lose any files when instance restarts
               Specify the output path on an Amazon S3 bucket where the trained model will persist
     2. Inference
            1. Neo + AWS IoT Greengrass = compiler + runtime = Edge computing
            2. SageMaker Batch Transform - no need real time and batch a lot
            3. Distributed GPU training = Horovod, Accelerated Computing / AI – GPU instances -> g3, g4, p2, p3
            4. Autopilot -> HPO / Ensembling (10 modells) . Auto => >100MB, HPO and smaller = Ensembling
            5. Inference Pipelines can be used to make either real-time predictions or to process batch transforms
                - 2-15 containers
                - Use pre-built TensorFlow docker images provided by SageMaker to train and host models on SageMaker infrastructure
                 Connection
                       Connect to the SageMaker API or to the SageMaker Runtime through an interface endpoint in your Virtual Private Cloud (VPC)
                        Use AWS Key Management Service (AWS KMS) to manage encryption keys for encrypting data at rest and use TLS for encrypting data in transit.
                        Enable network isolation for training jobs and models
            6. Elastic Inference = At fraction of cost of using a GPU instance  
   
   3. Monitoring
       1. Cloudwatch = system, cloudTrail = api, user activity
            SageMaker monitoring metrics are available on CloudWatch at a 1-minute frequency
         CloudWatch keeps the SageMaker monitoring statistics for 15 months. However, the Amazon CloudWatch console limits the search to metrics that were updated in the last 2 weeks
      2.  CloudTrail
               log user activities
               provides a record of actions taken by a user, role, or an AWS service in Amazon SageMaker. CloudTrail keeps this record for a period of 90 days

       
