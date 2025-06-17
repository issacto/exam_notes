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
   5. Datapipeline
   6. Quickinsight
