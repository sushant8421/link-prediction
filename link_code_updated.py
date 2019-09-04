import pandas as pd
import numpy as np
import networkx as nx
import sys
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
\
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

count_yes=0
num_of_nodes = 1000
  #list of yes edges in the network

def load_data():
	reduced_file = open('yes_edges.txt', 'w')
	count=0
	data = np.zeros([num_of_nodes,num_of_nodes])
	data_file_path = "/Users/Ashwin/Desktop/IITG/Sem 1/CS529/SNA1/Final Data/DBLP.txt"
	with open(data_file_path, "r") as ins:
		for line in ins:
			nodes = line.split() #splitting given file
			vertex1 = int(nodes[0])
			vertex2 = int(nodes[1])
			if vertex1 <= 1000 and vertex2 <= 1000:
				reduced_file.write(nodes[0]+" "+nodes[1]+"\n") #writing reduced data to reduced.txt file
				data[vertex1-1][vertex2-1] = 1
				count += 1


	data = np.array(data) #string dataframe in pandas to numpy array
	data = data.astype("int") #float64 to int64
	return data,count

data,count_yes=load_data()

length = 1000
A = np.array(data) #matrix to np array
G=nx.from_numpy_matrix(A) #making graph
#A = ([(length*length),['edge',0,0,0]])
A = [['edge',0,0,0]]
d =1

common_data = np.zeros([length,length])
jaccard_data = np.zeros([length,length])
adar_data = np.zeros([length,length])
pref_data = np.zeros([length,length])
resource_data = np.zeros([length,length])


common_data_no = np.zeros([length,length])
jaccard_data_no = np.zeros([length,length])
adar_data_no = np.zeros([length,length])
pref_data_no = np.zeros([length,length])
resource_data_no = np.zeros([length,length])

mean = np.zeros([5,2])                 #mean is an array of 5 features each for lable 1 and 0
variance = np.zeros([5,2])

naive_precision=[]
naive_recall_arr=[]
i=0;

def common_neighbour(data, length, G):
	a = []
	test = np.array(data)
	for i in range(length):
		for j in range(length):
			if i==j:
				common_data[i][j]=0

			else:
				and_op = np.bitwise_and(test[i,:], test[j,:])
				common_data[i][j]=np.count_nonzero(and_op)
			print(sorted(nx.common_neighbors(G,i,j)))


def jaccard(data, length, G):

	test = np.array(data)
	for i in range(length):
		for j in range(length):
			if i==j:
				jaccard_data[i][j]=0
			else:
				and_op = np.bitwise_and(test[i,:], test[j,:])
				or_op = np.bitwise_or(test[i,:], test[j,:])
				and_count=np.count_nonzero(and_op)
				or_count=np.count_nonzero(or_op)
				if or_count!=0:
					jaccard_data[i][j] = (and_count/or_count)
				else:
					jaccard_data[i][j] = 0

def adar(data, length, G):

	test = np.array(data)
	for i in range(length):
		for j in range(length):
			if i==j:
				adar_data[i][j]=0
			else:
				and_op = np.bitwise_and(test[i,:], test[j,:])
				and_count=np.count_nonzero(and_op)
				if and_count!=0:
					log_and = math.log(and_count)
					if log_and!=0:
						adar_data[i][j] = (1/log_and)
					else:
						adar_data[i][j] = 0
				else:
					adar_data[i][j] = 0

def pref_attach(data, length, G):
	test = np.array(data)
	for i in range(length):
		for j in range(length):
			if i==j:
				pref_data[i][j]=0
			else:

				n1 = np.count_nonzero(test[i,:])
				n2 = np.count_nonzero(test[j,:])
				pref_data[i][j]=(n1*n2)

def resource_alloc(data, length, G):

	test = np.array(data)
	for i in range(length):
		for j in range(length):
			if i==j:
				resource_data[i][j]=0
			else:
				and_op = np.bitwise_and(test[i,:], test[j,:])
				and_count=np.count_nonzero(and_op)
				if and_count!=0:
					resource_data[i][j] = (1/and_count)
				else:
					resource_data[i][j] = 0

def common_neighbour_no(data, length, G):
	test = np.array(data)
	for i in range(length):
		for j in range(length):
			if i==j:
				common_data_no[i][j]=0

			else:
				and_op = np.bitwise_and(test[i,:], test[j,:])
				common_data_no[i][j]=np.count_nonzero(and_op)

def jaccard_no(data, length, G):

	test = np.array(data)
	for i in range(length):
		for j in range(length):
			if i==j:
				jaccard_data_no[i][j]=0
			else:
				and_op = np.bitwise_and(test[i,:], test[j,:])
				or_op = np.bitwise_or(test[i,:], test[j,:])
				and_count=np.count_nonzero(and_op)
				or_count=np.count_nonzero(or_op)
				if or_count!=0:
					jaccard_data_no[i][j] = (and_count/or_count)
				else:
					jaccard_data_no[i][j] = 0

def adar_no(data, length, G):

	test = np.array(data)
	for i in range(length):
		for j in range(length):
			if i==j:
				adar_data_no[i][j]=0
			else:
				and_op = np.bitwise_and(test[i,:], test[j,:])
				and_count=np.count_nonzero(and_op)
				if and_count!=0:
					log_and = math.log(and_count)
					if log_and!=0:
						adar_data_no[i][j] = (1/log_and)
					else:
						adar_data_no[i][j] = 0
				else:
					adar_data_no[i][j] = 0

def pref_attach_no(data, length, G):
	test = np.array(data)
	for i in range(length):
		for j in range(length):
			if i==j:
				pref_data_no[i][j]=0
			else:

				n1 = np.count_nonzero(test[i,:])
				n2 = np.count_nonzero(test[j,:])
				pref_data_no[i][j]=(n1*n2)

def resource_alloc_no(data, length, G):

	test = np.array(data)
	for i in range(length):
		for j in range(length):
			if i==j:
				resource_data[i][j]=0
			else:
				and_op = np.bitwise_and(test[i,:], test[j,:])
				and_count=np.count_nonzero(and_op)
				if and_count!=0:
					resource_data_no[i][j] = (1/and_count)
				else:
					resource_data_no[i][j] = 0

def make_vector_yes(length, common_data, jaccard_data, adar_data):
	d =1
	new_path = 'yes_no_vectors.txt'
	new_path1 = 'only_yes_vectors.txt'
	new_vectors = open(new_path,'w')
	new_vectors1 = open(new_path1,'w')
	data_file_path = "yes_edges.txt"
	with open(data_file_path, "r") as ins:
		for line in ins:
			nodes = line.split() #splitting given file
			vertex1 = int(nodes[0])
			vertex2 = int(nodes[1])
			new_vectors.write(str(common_data[vertex1][vertex2])+" "+str(jaccard_data[vertex1][vertex2])+" "+str(adar_data[vertex1][vertex2])+" "+str(pref_data[vertex1][vertex2])+" "+str(resource_data[vertex1][vertex2])+" "+"1"+"\n")
			new_vectors1.write(str(common_data[vertex1][vertex2])+" "+str(jaccard_data[vertex1][vertex2])+" "+str(adar_data[vertex1][vertex2])+" "+str(pref_data[vertex1][vertex2])+" "+str(resource_data[vertex1][vertex2])+" "+"1"+"\n")

def make_no_file(count_yes):
	counter =0
	data_nonedges =np.zeros([num_of_nodes,num_of_nodes], dtype = int)
	no_file = open('no_edges.txt', 'w')

	while counter<=count_yes:
		a = random.randint(0,length-1)
		b = random.randint(0,length-1)
		if (a!=b) & data[a][b]!=1 & data_nonedges[a][b]==0:
			data_nonedges[a][b] = 1
			counter+=1
			no_file.write(str(a)+" "+str(b)+"\n")
			if counter>=count_yes:
				break
	return data_nonedges

def make_vector_no():
	new_path = 'yes_no_vectors.txt'
	new_path1 = 'only_no_vectors.txt'
	new_vectors = open(new_path,'a')
	new_vectors1 = open(new_path1,'w')
	data_file_path = "no_edges.txt"
	with open(data_file_path, "r") as ins:
		for line in ins:
			nodes = line.split() #splitting given file
			vertex1 = int(nodes[0])
			vertex2 = int(nodes[1])
			new_vectors.write(str(common_data_no[vertex1][vertex2])+" "+str(jaccard_data_no[vertex1][vertex2])+" "+str(adar_data_no[vertex1][vertex2])+" "+str(pref_data_no[vertex1][vertex2])+" "+str(resource_data_no[vertex1][vertex2])+" "+"0"+"\n")
			new_vectors1.write(str(common_data_no[vertex1][vertex2])+" "+str(jaccard_data_no[vertex1][vertex2])+" "+str(adar_data_no[vertex1][vertex2])+" "+str(pref_data_no[vertex1][vertex2])+" "+str(resource_data_no[vertex1][vertex2])+" "+"0"+"\n")

def find_m_v(x_train, y_train, count_yes):
	len_y = len(y_train)



	for i in range(len_y):
		if y_train.iloc[i] == 1:                    #finidng mean for each feature and for 0 and 1
			mean[0,0]+= x_train.iloc[i,0]           #by adding those values to 1 whose label is 1 and also same for 0
			mean[1,0]+= x_train.iloc[i,1]
			mean[2,0]+= x_train.iloc[i,2]
			mean[3,0]+= x_train.iloc[i,3]
			mean[4,0]+= x_train.iloc[i,4]
		else:
			mean[0,1]+= x_train.iloc[i,0]
			mean[1,1]+= x_train.iloc[i,1]
			mean[2,1]+= x_train.iloc[i,2]
			mean[3,1]+= x_train.iloc[i,3]
			mean[4,1]+= x_train.iloc[i,4]

	mean[0,0]/=count_yes
	mean[1,0]/=count_yes
	mean[2,0]/=count_yes
	mean[3,0]/=count_yes
	mean[4,0]/=count_yes

	mean[0,1]/=count_yes
	mean[1,1]/=count_yes
	mean[2,1]/=count_yes
	mean[3,1]/=count_yes
	mean[4,1]/=count_yes

	for i in range(len_y):
		if y_train.iloc[i]==1:
			variance[0,0]+= ((x_train.iloc[i,0] - mean[0,0])*(x_train.iloc[i,0] - mean[0,0]))

			variance[1,0]+= (x_train.iloc[i,1] - mean[1,0])*(x_train.iloc[i,1] - mean[1,0])

			variance[2,0]+= (x_train.iloc[i,2] - mean[2,0])*(x_train.iloc[i,2] - mean[2,0])  #subtracting mean of feature1 with lables 1 from each value with feature 1

			variance[3,0]+= (x_train.iloc[i,3] - mean[3,0])*(x_train.iloc[i,3] - mean[3,0])

			variance[4,0]+= (x_train.iloc[i,4] - mean[4,0])*(x_train.iloc[i,4] - mean[4,0])

		else:
			variance[0,1]+= (x_train.iloc[i,0] - mean[0,1])*(x_train.iloc[i,0] - mean[0,1])  #subtracting mean of feature1 with lables 1 from each value with feature 1

			variance[1,1]+= (x_train.iloc[i,1] - mean[1,1])*(x_train.iloc[i,1] - mean[1,1])

			variance[2,1]+= (x_train.iloc[i,2] - mean[2,1])*(x_train.iloc[i,2] - mean[2,1])  #subtracting mean of feature1 with lables 1 from each value with feature 1

			variance[3,1]+= (x_train.iloc[i,3] - mean[3,1])*(x_train.iloc[i,3] - mean[3,1])

			variance[4,1]+= (x_train.iloc[i,4] - mean[4,1])*(x_train.iloc[i,4] - mean[4,1])


	variance[0,0]/=count_yes
	variance[1,0]/=count_yes
	variance[2,0]/=count_yes
	variance[3,0]/=count_yes
	variance[4,0]/=count_yes

	variance[0,1]/=count_yes
	variance[1,1]/=count_yes
	variance[2,1]/=count_yes
	variance[3,1]/=count_yes
	variance[4,1]/=count_yes

def gaussian_pred(predicted_values, x_train, y_train, x_test, y_test):

	fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted_values, pos_label=1)
	auc = metrics.auc(fpr, tpr)
	precision = precision_score(y_test, predicted_values, average='micro')
	recall = recall_score(y_test, predicted_values, average='micro')

	return auc, precision,recall,fpr,tpr



def prediction(count_yes, length):

	vector = pd.read_csv("yes_no_vectors.txt", sep = " ")
	x = vector.iloc[:, 0:5]
	y = vector.iloc[:, -1]

	kf = KFold(n_splits = 5, random_state = None, shuffle = True)

	naive_auc=naive_preci=naive_recall=naive_fpr=naive_tpr=0
	svm_auc=svm_preci=svm_recall=svm_fpr=svm_tpr=0
	decision_auc=decision_preci=decision_recall=d_fpr=d_tpr=0


	for train_index, test_index in kf.split(x):
		x_train, x_test = x.iloc[train_index], x.iloc[test_index]      #splitting the dataset into train and test
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]
		find_m_v(x_train, y_train, count_yes)
		predicted_values = np.zeros([len(x_test)], dtype=int)
		for i in range (len(x_test)):
			p1=0
			p2=0
			for j in range(5):
				den_temp = math.sqrt(2*(math.pi)*variance[j][0])
				power_mean = -((x_test.iloc[i,j] - ((mean[j][0])*(mean[j][0])))/(2*((variance[j][0]))))
				naive_b_one = math.pow(math.e,power_mean)
				naive_b_one = (1/den_temp)*naive_b_one
				p1+=naive_b_one
				#finding naive bayes for label 1 and label 0
				#by calculating the same for each feature of the test dataset


				den_temp_zero = math.sqrt(2*(math.pi)*variance[j][1])
				power_mean_zero = -((x_test.iloc[i,j] - ((mean[j][1])*(mean[j][1])))/(2*((variance[j][1]))))
				naive_b_zero = math.pow(math.e,power_mean_zero)
				naive_b_zero = (1/den_temp_zero)*naive_b_zero
				p2+=naive_b_zero
				if p1>p2:
					predicted_values[i] = 1
				else:
					predicted_values[i] = 0
			#print(predicted_values)
		a,b,c,p,q=gaussian_pred(predicted_values, x_train, y_train, x_test, y_test)
		naive_auc+=a
		naive_preci+=b
		naive_recall+=c
		naive_fpr+=p
		naive_tpr+=q
		


	lw = 2
	plt.plot(naive_fpr/5, naive_tpr/5, color='darkorange',
	lw=lw, label='ROC curve Naive Bayes (area = %0.2f)' % (float(naive_auc)/5))
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	label = mpatches.Patch(color='darkorange', label='ROC curve Naive Bayes (area = %0.2f)' % (float(naive_auc)/5))
	plt.legend(handles=[label])
	plt.show()



	print("Naive Bayes :\n",)
	print("AUC: ",naive_auc/5,"  Precision: ",naive_preci/5,"  Recall: ",naive_recall/5,"\n")
	


def topo_prediction(count_yes,length):                 #function for 1/5 YES and all NO calculating topological
		vector = pd.read_csv("only_yes_vectors.txt", sep = " ")
		x = vector.iloc[:, 0:5]
		y = vector.iloc[:, -1]

		vector1 = pd.read_csv("only_no_vectors.txt", sep=" ")
		x1 = vector.iloc[:, 0:5]
		y1 = vector.iloc[:, -1]

		kf = KFold(n_splits = 5, random_state = None, shuffle = True)

		for train_index, test_index in kf.split(x):
			x_train, x_test = x.iloc[train_index], x.iloc[test_index]
			y_train, y_test = y.iloc[train_index], y.iloc[test_index]



		#test contains 1/5th of dataset
		#combining 1/5th of Yes with all No





		frames_x = [x_test,x1]
		frames_y = [y_test,y1]


		x_test_set = pd.concat(frames_x)
		y_test_set = pd.concat(frames_y)


		#Average AUC CN

		n1 =0
		for i in range (len(x_test)):
			for j in range(len(x1)):
				if (x_test.iloc[i,0] > x1.iloc[j,0]):
					n1+=1

		n2 =0
		for i in range (len(x_test)):
			for j in range(len(x1)):
				if (x_test.iloc[i,0] == x1.iloc[j,0]):
					n2+=1


		avg_auc_CN = (n1+((1/2)*n2))/((len(x_test))*(len(x1)))

		print("Average AUC Common Neighbour",avg_auc_CN,"\n")

		p = x_train.iloc[:, 0:1]

		q = p.nlargest(2,'0.0',keep='last')

		#print("Highest Common Neigbour Score: ",q.iloc[0,0],"\n")




		#Average AUC JC
		n1 =0
		for i in range (len(x_test)):
			for j in range(len(x1)):
				if (x_test.iloc[i,1] > x1.iloc[j,1]):
					n1+=1

		n2 =0
		for i in range (len(x_test)):
			for j in range(len(x1)):
				if (x_test.iloc[i,1] == x1.iloc[j,1]):
					n2+=1


		avg_auc_JC = (n1+((1/2)*n2))/((len(x_test))*(len(x1)))

		print("Average AUC Jaccard Coefficient",avg_auc_JC,"\n")

		p = x_train.iloc[:, 1:2]

		t = p.nlargest(2,'0.0.1',keep='last')

		#print("Highest Jaccard Coeff Score: ",t.iloc[0,0],"\n")






		#Average AUC AA
		n1 =0
		for i in range (len(x_test)):
			for j in range(len(x1)):
				if (x_test.iloc[i,2] > x1.iloc[j,2]):
					n1+=1

		n2 =0
		for i in range (len(x_test)):
			for j in range(len(x1)):
				if (x_test.iloc[i,2] == x1.iloc[j,2]):
					n2+=1


		avg_auc_AA = (n1+((1/2)*n2))/((len(x_test))*(len(x1)))

		print("Average AUC Adamic Adar",avg_auc_AA,"\n")

		p = x_train.iloc[:, 2:3]

		q = p.nlargest(2,'0.0.2',keep='last')

		#print("Highest Adamic Adar Score: ",q.iloc[0,0],"\n")






		#Average AUC Preferential Attachment

		n1 =0
		for i in range (len(x_test)):
			for j in range(len(x1)):
				if (x_test.iloc[i,3] > x1.iloc[j,3]):
					n1+=1

		n2 =0
		for i in range (len(x_test)):
			for j in range(len(x1)):
				if (x_test.iloc[i,3] == x1.iloc[j,3]):
					n2+=1


		avg_auc_PA = (n1+((1/2)*n2))/((len(x_test))*(len(x1)))

		print("Average AUC Preferential Attachment",avg_auc_PA,"\n")

		p = x_train.iloc[:, 3:4]

		q = p.nlargest(2,'0.0.3',keep='last')

		#print("Highest Preferntial Attachment Score: ",q.iloc[0,0],"\n")






		#Average AUC RA
		n1 =0
		for i in range (len(x_test)):
			for j in range(len(x1)):
				if (x_test.iloc[i,4] > x1.iloc[j,4]):
					n1+=1

		n2 =0
		for i in range (len(x_test)):
			for j in range(len(x1)):
				if (x_test.iloc[i,4] == x1.iloc[j,4]):
					n2+=1


		avg_auc_RA = (n1+((1/2)*n2))/((len(x_test))*(len(x1)))

		print("Average AUC Resource Allocation",avg_auc_RA,"\n")



		p = x_train.iloc[:, 4:5]

		q = p.nlargest(2,'0.0.4',keep='last')

		#print("Highest Resource Allocation Score: ",q.iloc[0,0],"\n")




#common_neighbour(data, length, G)
#jaccard(data, length, G)
# adar(data, length, G)
# pref_attach(data, length, G)
# resource_alloc(data, length, G)
# data_nonedges = make_no_file(count_yes)
# common_neighbour_no(data_nonedges, length, G)
# jaccard_no(data_nonedges, length, G)
# adar_no(data_nonedges, length, G)
# pref_attach_no(data_nonedges, length, G)
# resource_alloc_no(data_nonedges, length, G)
# make_vector_yes(length, common_data, jaccard_data, adar_data)
# make_no_file(count_yes)
# make_vector_no()
#prediction(count_yes, length)
topo_prediction(count_yes,length)
