# ISL
data preprocess or generation
1 data_generation_syn
2 data_generation_real
output diferent environment

ISL supervise
model function: defination nolinear...
utils : save dag, draw, rank w, accuracy... K-means

one main,py read multi env data, call model, call utils --> DAG, Y-related DAG

strategy (1 keep all, 2 other) threshold as function in util

predictor of Y, MLP

ISL unsupervise

aggregate (manually), retrain NOTEAR-MLP
(backup), load /theta^{Y}_{1}, initialize / fix













nonlinear.py is the original notear-mlp. 

sachs.csv is the preprocessed sachs data by removing the header for easier loadind with numpy.loadtxt

$ utils.py $ includes the data generation, plotting, clustering for different environment. 


$ nonlinear\_ourDataNonlinear\_daring.py $ is for testing notear on data generated in DARING's manner nonlinearly. 


$ notearMLP\_experiment\_enviroonment.py $ is for testing different environments. 


$ nonlinear\_castle $ is the modified notear-mlp with addtional y prediction loss added for real data. y_index in notear-mlp indicates which feature is treated as y 


$ sachs\_data\_NOTEAR-MLP.py $ is for testing notear and its modificaitons on real data set (sachs protein data)
usage: 
<pre><code> 
python sachs_data_NOTEAR−MLP.py sachs.csv
</code></pre>
To switch to a differnt version of notear, modify ...
<pre><code> 
import ... as nonlinear
</code></pre>
To use a different feature as label $y$, change the $y\_index$ in $notears\_nonlinear$


$ sachs\_cluster $ is for clustering the sachs as different environment and applying notear-mlp on them. 

usage: 
<pre><code> 
python sachs_cluster.py sachs.csv</code></pre>
