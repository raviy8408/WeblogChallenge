# WeblogChallenge
This is an interview challenge for Paytm Labs. Please feel free to fork. Pull Requests will be ignored.

The challenge is to make make analytical observations about the data using the distributed tools below.

## Section 1:  Processing & Analytical goals:

1. Sessionize the web log by IP. Sessionize = aggregrate all page hits by visitor/IP during a fixed time window.

2. Determine the average session time

3. Determine unique URL visits per session. To clarify, count a hit to a unique URL only once per session.

4. Find the most engaged users, ie the IPs with the longest session times

## Solution Section 1:
[solution_section_1.py](https://github.com/raviy8408/WeblogChallenge/blob/master/solution_section_1.py)

1. For any IP a session is defined as the gap between two consecutive requests should be less than 30 min. Column
"sessionized" is added to indicate the session of each request.
2. Code will print Descriptive statistics of "session time = session end time - session start time".
3. URL is extracted from request. The table displays the unique URl visit per session.
4. Average session time is calculated for all the IPs. The table is displayed in descending order of avg session time.

## Section 2: Additional questions for Machine Learning Engineer (MLE) candidates:

1. Predict the expected load (requests/second) in the next minute

### Solution:
[solution_section_2_q_1.py](https://github.com/raviy8408/WeblogChallenge/blob/master/solution_section_2_q_1.py)

Modeling Approach:
- Problem is solved using regression method. Although the problem could have also been solved using time series modeling,
however due to unavailability of any scalable time series methods regression is chosen.
- As data available was only for last few hours, variable are created using recent history. For
example total load in last 1, 15, 60 and 120 min is created as variable. Similar approach has been taken for all other
variables as well.
Variable list:
- last 1, 15, 60, 120 min total load
- last 1, 15, 60, 120 min sum of request_processing_time, backend_processing_time, response_processing_time and count of request type
- last 1, 15 min sum of elb_status_code and backend_status_code
- last 1, 15 min sum of received bytes and sent bytes

2. Predict the session length for a given IP

### Solution:
[solution_section_2_q_2.py](https://github.com/raviy8408/WeblogChallenge/blob/master/solution_section_2_q_2.py)

Modeling Approach:
- Regression based generalized model is build for all the IPs to predict the length of any session.
Variable list:
- distinct URL visit count for the IP
- session start hour as a categorical variable
- last 15 min average of request_processing_time, backend_processing_time, response_processing_time and count of request types
- last 15 min count of different elb_status_codes and backend_status_code
- last 15 min sum of received bytes and sent bytes

3. Predict the number of unique URL visits by a given IP

### Solution:
[solution_section_2_q_3.py](https://github.com/raviy8408/WeblogChallenge/blob/master/solution_section_2_q_3.py)

Modeling Approach:
- Regression model is build to predict URL visit count for IPs based of the uses data available. However user
demographic information would have made more sense for unique URL visit count prediction. As with the current model we
can only make prediction for existing users.
Variable list:
- Sum of session_time, received bytes, sent bytes, request_processing_time, backend_processing_time and response_processing_time per IP

### Solution section 2 general note:

- Features are created in multiple steps. At each step features are vectorized and finally all the features are assembled
together to create final feature vector.

- All the continuous features are scaled.

- Train and test data is splitted in 70:30 ratio.

- GBTRegression model is used to generate the prediction.

- RMSE obtained on test data will print at the bottom.

### Solution Limitations:
- Current solution does not focus on accuracy
- Focus has been on implementing scalable solution
- Given better computational resource following can be done to improve the accuracy:
    - Several new features can be added
    - Implement variable dimensionality reduction techniques
    - Cross validation for the model to identify best parameter set
    - Try out different model to asses their performance


#######################################################################################################################


## Tools allowed (in no particular order):
- Spark (any language, but prefer Scala or Java)
- Pig
- MapReduce (Hadoop 2.x only)
- Flink
- Cascading, Cascalog, or Scalding

If you need Hadoop, we suggest 
HDP Sandbox:
http://hortonworks.com/hdp/downloads/
or 
CDH QuickStart VM:
http://www.cloudera.com/content/cloudera/en/downloads.html


### Additional notes:
- You are allowed to use whatever libraries/parsers/solutions you can find provided you can explain the functions you are implementing in detail.
- IP addresses do not guarantee distinct users, but this is the limitation of the data. As a bonus, consider what additional data would help make better analytical conclusions
- For this dataset, complete the sessionization by time window rather than navigation. Feel free to determine the best session window time on your own, or start with 15 minutes.
- The log file was taken from an AWS Elastic Load Balancer:
http://docs.aws.amazon.com/ElasticLoadBalancing/latest/DeveloperGuide/access-log-collection.html#access-log-entry-format



## How to complete this challenge:

A. Fork this repo in github
    https://github.com/PaytmLabs/WeblogChallenge

B. Complete the processing and analytics as defined first to the best of your ability with the time provided.

C. Place notes in your code to help with clarity where appropriate. Make it readable enough to present to the Paytm Labs interview team.

D. Complete your work in your own github repo and send the results to us and/or present them during your interview.

## What are we looking for? What does this prove?

We want to see how you handle:
- New technologies and frameworks
- Messy (ie real) data
- Understanding data transformation
This is not a pass or fail test, we want to hear about your challenges and your successes with this particular problem.
