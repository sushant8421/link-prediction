# link-prediction
To predict which new connections are likely to occur soon(Friend Suggestion).

Link prediction is useful in the network to predict which new conections are likely to occur soon or in which subnetwork any new news or concept is to be shared so that it will spread rapidly to the whole network. 

According to assignment we have implemented 5 types of topological network-based link prediction methods listed as:

    • Common Neighbour (CN)
    • Jaccard Coefficient (JC)
    • Adamic Adar (AA)
    • Resource Allocation(RA)
    • Preferential Attachment (PA)
    
We considered the above features calculated for each edge as input features for the below 3 classification methods and gave each edge a label (1/0) depending on whether the edge is present in the network or not.
The implemented classification techniques for link prediction are:

    • Naïve Bayes

Used 5-fold cross validation for finding precision, recall and AUC scores for the classifiers.
Libraries used: 

    • Pandas
    • NumPy
    • Networkx
    • Matplotlib
    • Scikitlearn


#Detailed description:

    1. For performing link prediction methods, we made matrix of all currently present edges in the network and found scores        of each technique as follows:
    
        ◦ Common Neighbours: 
            ▪ Score(x,y) = | N(x) intersection N(y) |
            
        ◦ Jaccard Coefficient: 
            ▪ Score(x,y) = | N(x) intersection N(y) / N(x) union N(x) |
            
        ◦ Adamic Adar:  
            ▪ Score(x,y) = 1/Log(| N(x) intersection N(y) |)
            
        ◦ Resource allocation:
            ▪ Score(x,y) = 1/(| N(x) intersection N(y) |)
            
        ◦ Preferential Attachment:
            ▪ Score(x,y) = N(x) * N(y)
            
        ◦ Procedure:
            i. Randomly selected equal number of absent edges (NO edges) as the number of present edges in the network.
            ii. For validation, we have used one-fold of the YES edges and entire NO edges as test dataset and four-fold of                   YES edges as train dataset.
            iii. If the score of the test vector is lesser than the scores of all train vectors, then we predict it as label                  0, else we predict it as label 1.

    2. We have implemented naïve bayes classifier from scratch.
    
        ◦ Procedure: 
        
            i. Randomly selected equal number of absent edges as the number of present edges in the network.
            ii. Created a vector for each edge with five features explained above and label for each edge as:  if it is                       present in the network, then label it as 1 otherwise 0.
            iii. Implemented Gaussian naïve bayes since input features are continuous.
            iv. Calculated mean and variance for each feature with respect to label 1 and label 0.
            v. Then, we used Gaussian function to predict the probabilities of the input vector belonging to label 0 or label                 1.
            vi. If p(label 1|x) > p(label 0|x), then we predict that the input vector belongs to label 1 i.e it is likely to                  become an edge in the network soon.
            vii. Else, we predict label 0 i.e the connection is unlikely in the future.
