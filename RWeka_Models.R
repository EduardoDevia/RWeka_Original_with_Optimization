#install.packages("doSNOW")

library(grid)
library(partykit)
library(RWeka)
library(partykit)
library(FSelector)
library(e1071)
library(lattice)
library(ggplot2)
library(caret)
library(RWekajars)
library(evaluate)
library(proto)
library(gsubfn)
library(DBI)
library(RSQLite)
library(DBI)
library(tcltk)
library(sqldf)
library(plyr)
library(dplyr)





#options(java.parameters = "-Xmx10000m")
memory.limit(80000)
memory.limit()
memory.size(TRUE)
memory.size()
memory.profile()

#help(make_Weka_associator)
#WOW("M5P")
#Train total rows

#==============Preprocessing====================
setwd("C:/Users/eduar/Documents")

#Load the arff file
train_file <- read.csv("train.csv", stringsAsFactors = FALSE)
test_file <- read.csv("test.csv", stringsAsFactors = FALSE)
#Check the file summary
summary(train_file)

#Replace -1 for standard NA it applies just for this file
train_file[train_file=="-1"]<-NA
test_file[test_file=="-1"]<-NA
#Remove NA values
train_file_clean<-na.omit(train_file)
test_file_clean<-na.omit(test_file)

#============Changing values to categorical ======================

#summary checking the type of line / factor/ int/ etc
glimpse(train_file)
#Change some fields to categorical
train_file$target<-as.factor(train_file$target)
train_file$ps_car_01_cat<-as.factor(train_file$ps_car_01_cat)
train_file$ps_car_02_cat<-as.factor(train_file$ps_car_02_cat)
train_file$ps_car_03_cat<-as.factor(train_file$ps_car_03_cat)
train_file$ps_car_04_cat<-as.factor(train_file$ps_car_04_cat)
train_file$ps_car_05_cat<-as.factor(train_file$ps_car_05_cat)
train_file$ps_car_06_cat<-as.factor(train_file$ps_car_06_cat)
train_file$ps_car_07_cat<-as.factor(train_file$ps_car_07_cat)
train_file$ps_car_08_cat<-as.factor(train_file$ps_car_08_cat)
train_file$ps_car_09_cat<-as.factor(train_file$ps_car_09_cat)
train_file$ps_car_10_cat<-as.factor(train_file$ps_car_10_cat)
train_file$ps_ind_02_cat <-as.factor(train_file$ps_ind_02_cat)
train_file$ps_ind_04_cat <-as.factor(train_file$ps_ind_05_cat)
train_file$ps_ind_05_cat <-as.factor(train_file$ps_ind_05_cat)
#============Resampling data ================

resample<-make_Weka_filter("weka/filters/supervised/instance/Resample")

train_file_clean<-resample(target~ .,data=train_file,control=Weka_control(Z=10))

#============Setting up variables for further calculations===================
#Count total records
Total_Records<-sqldf("Select count(*) from train_file_clean")
Total_Records<-Total_Records$`count(*)`
# Check sensibity and 
summary(train_file_clean$target)/(Total_Records)
summary(train_file_clean$target)/(573518+21694)


write.table(train_file_clean, file = "~/train_file_clean.csv", sep = ",", col.names = NA, qmethod = "double")   


#===============Building the Models===================
#Numeric
functions
MultilayerPerceptron
SMOreg
#IBk
Bagging
DecisionTable
M5Rules
ZeroR
DecisionStump
M5P
RamdonForest
REPTree
#================Classification==============#
#BayesNet
#naiveBayes
#Logistic
#MultilayerPerceptron
#SMO
#Bagging
#LogitBoost
DecistionTable
#OneR
#Part
#ZeroR
DesicionStump #---- RUN cv
#J48
#LMT
#randomForest         ----------Java.lang.OutOfMemoryError
#Randomtree
#REPTree

#===============ZeroR===================
ZeroR<-make_Weka_classifier("weka/classifiers/rules/ZeroR")
ZeroR_Classifier<-ZeroR(train_file_clean$target~ ., data = train_file_clean)
ZeroR_Train<-summary(ZeroR_Classifier)
#Cross Validation
ZeroR_CV <- evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
#===============OneR===================
OneR_Classifier<-OneR(train_file_clean$target~ ., data = train_file_clean)
OneR_Train<-summary(OneR_Classifier)
#Cross Validation
OneR_CV <- evaluate_Weka_classifier(OneR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
#===============MultiLayerPerceptron===================
MultilayerPerceptron<-make_Weka_classifier("weka/classifiers/functions/MultilayerPerceptron")
MultilayerPerceptron_Classifier<-MultilayerPerceptron(train_file_clean$target~ ., data = train_file_clean)
MultilayerPerceptron_Train<-summary(MultilayerPerceptron_Classifier)
#Cross Validation
MultilayerPerceptron_CV <- evaluate_Weka_classifier(MultilayerPerceptron_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
#===============J48===================
J48_Classifier<-J48(train_file_clean$target~ ., data = train_file_clean)
J48_Train<-summary(J48_Classifier)
#Cross Validation
J48_CV <- evaluate_Weka_classifier(J48_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
#===============IBk===================
IBk_Classifier<-IBk(train_file_clean$target~ ., data = train_file_clean,control=Weka_control(K=1))
IBK_Train<-summary(IBk_Classifier)
#Cross Validation
IBk_CV <- evaluate_Weka_classifier(IBk_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
#===============BayesNet===================
BayesNet<-make_Weka_classifier("weka/classifiers/bayes/BayesNet")
BayesNet_Classifier<-BayesNet(train_file_clean$target~ ., data = train_file_clean)
BayesNet_Train<-summary(BayesNet_Classifier)
#Cross Validation
BayesNet_CV <- evaluate_Weka_classifier(BayesNet_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
#===============NaiveBayes===================
NaiveBayes<-make_Weka_classifier("weka/classifiers/bayes/NaiveBayes")
NaiveBayes_Classifier<-NaiveBayes(train_file_clean$target~ ., data = train_file_clean)
NaiveBayes_Train<-summary(NaiveBayes_Classifier)
#Cross Validation
NaiveBayes_CV <- evaluate_Weka_classifier(NaiveBayes_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
#===============Logistic===================
Logistic_Classifier<-Logistic(train_file_clean$target~ ., data = train_file_clean)
Logistic_Train<-summary(Logistic_Classifier)
#Cross Validation
Logistic_CV <- evaluate_Weka_classifier(Logistic_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
#===============SMO===================
SMO_Classifier<-SMO(train_file_clean$target~ ., data = train_file_clean)
SMO_Train<-summary(SMO_Classifier)
#Cross Validation
SMO_CV <- evaluate_Weka_classifier(SMO_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
#===============LMT===================
LMT_Classifier<-LMT(train_file_clean$target~ ., data = train_file_clean)
LMT_Train<-summary(LMT_Classifier)
#Cross Validation
LMT_CV <- evaluate_Weka_classifier(LMT_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
#===============SMO===================
SMO_Classifier<-SMO(train_file_clean$target~ ., data = train_file_clean)
SMO_Train<-summary(SMO_Classifier)
#Cross Validation
SMO_CV <- evaluate_Weka_classifier(SMO_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
#===============RandomForest===================
RandomForest<-make_Weka_classifier("weka/classifiers/trees/RandomForest")
RandomForest_Classifier<-RandomForest(train_file_clean$target~ ., data = train_file_clean)
RandomForest_Train<-summary(RandomForest_Classifier)
#Cross Validation
RandomForest_CV <- evaluate_Weka_classifier(RandomForest_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
#===============RandomTree===================
RandomTree<-make_Weka_classifier("weka/classifiers/trees/RandomTree")
RandomTree_Classifier<-RandomTree(train_file_clean$target~ ., data = train_file_clean)
RandomTree_Train<-summary(RandomTree_Classifier)
#Cross Validation
RandomTree_CV <- evaluate_Weka_classifier(RandomTree_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
#===============REPTree===================
REPTree<-make_Weka_classifier("weka/classifiers/trees/REPTree")
REPTree_Classifier<-REPTree(train_file_clean$target~ ., data = train_file_clean)
REPTree_Train<-summary(REPTree_Classifier)
#Cross Validation
REPTree_CV <- evaluate_Weka_classifier(REPTree_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
#===============DecisionStump===================
DecisionStump_Classifier<-DecisionStump(train_file_clean$target~ ., data = train_file_clean)
DecisionStump_Train<-summary(DecisionStump_Classifier)
#Cross Validation
DecisionStump_CV <- evaluate_Weka_classifier(DecisionStump_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
#===============PART===================
PART_Classifier<-PART(train_file_clean$target~ ., data = train_file_clean)
PART_Train<-summary(PART_Classifier)
#Cross Validation
PART_CV <- evaluate_Weka_classifier(PART_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)

#=============Table Models to choose Bagging model ==============
Models<-c("ZeroR","OneR","BayesNet","DecisionStump","IBK","J48","LMT","Logistic","MultilayerPerceptron","NaiveBayes","PART","RandomForest","RandomTree","REPTree","SMO")
Accuracy<-c(ZeroR_true_Accuracy,OneR_true_Accuracy,BayesNet_true_Accuracy,DecisionStump_true_Accuracy,IBK_true_Accuracy,J48_true_Accuracy,LMT_true_Accuracy,Logistic_true_Accuracy,MultilayerPerceptron_true_Accuracy,NaiveBayes_true_Accuracy,PART_true_Accuracy,RandomForest_true_Accuracy,RandomTree_true_Accuracy,REPTree_true_Accuracy,SMO_true_Accuracy)
Cross_Val_Accuracy<-c(ZeroR_true_Accuracy_CV,OneR_Accuracy_CV,BayesNet_true_Accuracy_CV,DecisionStump_true_Accuracy_CV,IBK_true_Accuracy_CV,J48_true_Accuracy_CV,LMT_true_Accuracy_CV,Logistic_true_Accuracy_CV,MultilayerPerceptron_true_Accuracy_CV,NaiveBayes_true_Accuracy_CV,PART_true_Accuracy_CV,RandomForest_true_Accuracy_CV,RandomTree_true_Accuracy_CV,REPTree_true_Accuracy_CV,SMO_true_Accuracy_CV)
Yes_Correct_Clasified<-c(ZeroR_Train$confusionMatrix[2,2],OneR_Train$confusionMatrix[2,2],BayesNet_Train$confusionMatrix[2,2],DecisionStump_Train$confusionMatrix[2,2],IBK_Train$confusionMatrix[2,2],J48_Train$confusionMatrix[2,2],LMT_Train$confusionMatrix[2,2],Logistic_Train$confusionMatrix[2,2],MultilayerPerceptron_Train$confusionMatrix[2,2],NaiveBayes_Train$confusionMatrix[2,2],PART_Train$confusionMatrix[2,2],RandomForest_Train$confusionMatrix[2,2],RandomTree_Train$confusionMatrix[2,2],REPTree_Train$confusionMatrix[2,2],SMO_Train$confusionMatrix[2,2])
Yes_Correct_Clasified_CV<-c(ZeroR_CV$confusionMatrix[2,2],OneR_CV$confusionMatrix[2,2],BayesNet_CV$confusionMatrix[2,2],DecisionStump_CV$confusionMatrix[2,2],IBK_CV$confusionMatrix[2,2],J48_CV$confusionMatrix[2,2] ,LMT_CV$confusionMatrix[2,2],Logistic_CV$confusionMatrix[2,2],MultilayerPerceptron_CV$confusionMatrix[2,2],NaiveBayes_CV$confusionMatrix[2,2],PART_CV$confusionMatrix[2,2],RandomForest_CV$confusionMatrix[2,2],RandomTree_CV$confusionMatrix[2,2],REPTree_CV$confusionMatrix[2,2],SMO_CV$confusionMatrix[2,2])
No_Correct_Clasified<-c(ZeroR_Train$confusionMatrix[1,1],OneR_Train$confusionMatrix[1,1],Bagging_Train$confusionMatrix[1,1],BayesNet_Train$confusionMatrix[1,1],DecisionStump_Train$confusionMatrix[1,1],IBK_Train$confusionMatrix[1,1],J48_Train$confusionMatrix[1,1],LMT_Train$confusionMatrix[1,1],Logistic_Train$confusionMatrix[1,1],MultilayerPerceptron_Train$confusionMatrix[1,1],NaiveBayes_Train$confusionMatrix[1,1],PART_Train$confusionMatrix[1,1],RandomForest_Train$confusionMatrix[1,1],RandomTree_Train$confusionMatrix[1,1],REPTree_Train$confusionMatrix[1,1],SMO_Train$confusionMatrix[1,1])
No_Correct_Clasified_CV<-c(ZeroR_CV$confusionMatrix[1,1],OneR_CV$confusionMatrix[1,1] ,Bagging_CV$confusionMatrix[1,1],BayesNet_CV$confusionMatrix[1,1],DecisionStump_CV$confusionMatrix[1,1],IBk_CV$confusionMatrix[1,1],J48_CV$confusionMatrix[1,1] ,LMT_CV$confusionMatrix[1,1],Logistic_CV$confusionMatrix[1,1],MultilayerPerceptron_CV$confusionMatrix[1,1],NaiveBayes_CV$confusionMatrix[1,1],PART_CV$confusionMatrix[1,1],RandomForest_CV$confusionMatrix[1,1],RandomTree_CV$confusionMatrix[1,1],REPTree_CV$confusionMatrix[1,1],SMO_CV$confusionMatrix[1,1])

Table_Models<-data.frame(Models,Accuracy,Cross_Val_Accuracy,Yes_Correct_Clasified,No_Correct_Clasified,Yes_Correct_Clasified_CV,No_Correct_Clasified_CV)
TN<-summary(train_file_clean$target)[2]#True Negative
TP<-summary(train_file_clean$target)[1]#True Positive
Table_Models$Accuracy<-(Yes_Correct_Clasified+No_Correct_Clasified)/(TN+TP)
Table_Models$Cross_Val_Accuracy<-(Yes_Correct_Clasified_CV+No_Correct_Clasified_CV)/(TN+TP)
Table_Models$Sensitivity<-(No_Correct_Clasified/TP)*100
Table_Models$Sensitivity_CV<-(No_Correct_Clasified_CV/TP)*100
Table_Models$Specificity<-(Yes_Correct_Clasified/TN)*100
Table_Models$Specificity_CV<-(Yes_Correct_Clasified_CV/TN)*100
Table_Models$Overfitting<-(Accuracy-Cross_Val_Accuracy)*100
Table_Models$Ensamble<-ifelse(Table_Models$Models=="ZeroR"|Table_Models$Models=="OneR",-1,0)
Table_Models<-Table_Models[order(Table_Models$Ensamble,Table_Models$Overfitting,Table_Models$Yes_Correct_Clasified_CV),]

#===============Bagging===================

Bagging_Classifier<-Bagging(train_file_clean$target~ ., data = train_file_clean, control = Weka_control(W="weka.classifiers.trees.RandomTree"), na.action=NULL)
Bagging_Train<-summary(Bagging_Classifier)
#Cross Validation
Bagging_CV <- evaluate_Weka_classifier(Bagging_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE, na.action=NULL)
WOW(Bagging)
#===============LogitBoost===================
LogitBoost_Classifier<-LogitBoost(train_file_clean$target~ ., data = train_file_clean,control = Weka_control(W="weka.classifiers.trees.RandomTree"))
LogitBoost_Train<-summary(LogitBoost_Classifier)
#Cross Validation
LogitBoost_CV <- evaluate_Weka_classifier(LogitBoost_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
#===============AdaBoostM1===================
AdaBoostM1_Classifier<-AdaBoostM1(train_file_clean$target~ ., data = train_file_clean,control = Weka_control(W="weka.classifiers.trees.RandomTree"))
AdaBoostM1_Train<-summary(AdaBoostM1_Classifier)
#Cross Validation
AdaBoostM1_CV <- evaluate_Weka_classifier(AdaBoostM1_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
WOW(MultiBoostAB)
#===============Stacking===================
Stacking_Classifier<-Stacking(train_file_clean$target~ ., data = train_file_clean,control = Weka_control(
  M="weka.classifiers.trees.RandomTree",
  B="weka.classifiers.trees.RandomForest"  ))
Stacking_Train<-summary(Stacking_Classifier)
#Cross Validation
Stacking_CV <- evaluate_Weka_classifier(Stacking_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)

#==============Joint Models=================
Models<-c("ZeroR","OneR","Bagging","BayesNet","DecisionStump","IBK","J48","LMT","Logistic","LogitBoost","MultilayerPerceptron","NaiveBayes","PART","RandomForest","RandomTree","REPTree","SMO","AdaBoostM1","Stacking")
#Accuracy<-c(ZeroR_true_Accuracy,OneR_true_Accuracy,Bagging_true_Accuracy,BayesNet_true_Accuracy,DecisionStump_true_Accuracy,IBK_true_Accuracy,J48_true_Accuracy,LMT_true_Accuracy,Logistic_true_Accuracy,LogitBoost_true_Accuracy,MultilayerPerceptron_true_Accuracy,NaiveBayes_true_Accuracy,PART_true_Accuracy,RandomForest_true_Accuracy,RandomTree_true_Accuracy,REPTree_true_Accuracy,SMO_true_Accuracy,AdaBoostM1_true_Accuracy,Stacking_true_Accuracy)
#Cross_Val_Accuracy<-c(ZeroR_true_Accuracy_CV,OneR_Accuracy_CV,Bagging_true_Accuracy_CV,BayesNet_true_Accuracy_CV,DecisionStump_true_Accuracy_CV,IBK_true_Accuracy_CV,J48_true_Accuracy_CV,LMT_true_Accuracy_CV,Logistic_true_Accuracy_CV,LogitBoost_true_Accuracy_CV,MultilayerPerceptron_true_Accuracy_CV,NaiveBayes_true_Accuracy_CV,PART_true_Accuracy_CV,RandomForest_true_Accuracy_CV,RandomTree_true_Accuracy_CV,REPTree_true_Accuracy_CV,SMO_true_Accuracy_CV,AdaBoostM1_true_Accuracy_CV,Stacking_true_Accuracy_CV)
Yes_Correct_Clasified<-c(ZeroR_Train$confusionMatrix[2,2],OneR_Train$confusionMatrix[2,2],Bagging_Train$confusionMatrix[2,2],BayesNet_Train$confusionMatrix[2,2],DecisionStump_Train$confusionMatrix[2,2],IBK_Train$confusionMatrix[2,2],J48_Train$confusionMatrix[2,2],LMT_Train$confusionMatrix[2,2],Logistic_Train$confusionMatrix[2,2],LogitBoost_Train$confusionMatrix[2,2],MultilayerPerceptron_Train$confusionMatrix[2,2],NaiveBayes_Train$confusionMatrix[2,2],PART_Train$confusionMatrix[2,2],RandomForest_Train$confusionMatrix[2,2],RandomTree_Train$confusionMatrix[2,2],REPTree_Train$confusionMatrix[2,2],SMO_Train$confusionMatrix[2,2],AdaBoostM1_Train$confusionMatrix[2,2],Stacking_Train$confusionMatrix[2,2])
Yes_Correct_Clasified_CV<-c(ZeroR_CV$confusionMatrix[2,2],OneR_CV$confusionMatrix[2,2] ,Bagging_CV$confusionMatrix[2,2],BayesNet_CV$confusionMatrix[2,2],DecisionStump_CV$confusionMatrix[2,2],IBk_CV$confusionMatrix[2,2],J48_CV$confusionMatrix[2,2] ,LMT_CV$confusionMatrix[2,2],Logistic_CV$confusionMatrix[2,2],LogitBoost_CV$confusionMatrix[2,2],MultilayerPerceptron_CV$confusionMatrix[2,2],NaiveBayes_CV$confusionMatrix[2,2],PART_CV$confusionMatrix[2,2],RandomForest_CV$confusionMatrix[2,2],RandomTree_CV$confusionMatrix[2,2],REPTree_CV$confusionMatrix[2,2],SMO_CV$confusionMatrix[2,2],AdaBoostM1_CV$confusionMatrix[2,2],Stacking_CV$confusionMatrix[2,2])
No_Correct_Clasified<-c(ZeroR_Train$confusionMatrix[1,1],OneR_Train$confusionMatrix[1,1],Bagging_Train$confusionMatrix[1,1],BayesNet_Train$confusionMatrix[1,1],DecisionStump_Train$confusionMatrix[1,1],IBK_Train$confusionMatrix[1,1],J48_Train$confusionMatrix[1,1],LMT_Train$confusionMatrix[1,1],Logistic_Train$confusionMatrix[1,1],LogitBoost_Train$confusionMatrix[1,1],MultilayerPerceptron_Train$confusionMatrix[1,1],NaiveBayes_Train$confusionMatrix[1,1],PART_Train$confusionMatrix[1,1],RandomForest_Train$confusionMatrix[1,1],RandomTree_Train$confusionMatrix[1,1],REPTree_Train$confusionMatrix[1,1],SMO_Train$confusionMatrix[1,1],AdaBoostM1_Train$confusionMatrix[1,1],Stacking_Train$confusionMatrix[1,1])
No_Correct_Clasified_CV<-c(ZeroR_CV$confusionMatrix[1,1],OneR_CV$confusionMatrix[1,1] ,Bagging_CV$confusionMatrix[1,1],BayesNet_CV$confusionMatrix[1,1],DecisionStump_CV$confusionMatrix[1,1],IBk_CV$confusionMatrix[1,1],J48_CV$confusionMatrix[1,1] ,LMT_CV$confusionMatrix[1,1],Logistic_CV$confusionMatrix[1,1],LogitBoost_CV$confusionMatrix[1,1],MultilayerPerceptron_CV$confusionMatrix[1,1],NaiveBayes_CV$confusionMatrix[1,1],PART_CV$confusionMatrix[1,1],RandomForest_CV$confusionMatrix[1,1],RandomTree_CV$confusionMatrix[1,1],REPTree_CV$confusionMatrix[1,1],SMO_CV$confusionMatrix[1,1],AdaBoostM1_CV$confusionMatrix[1,1],Stacking_CV$confusionMatrix[1,1])
Table_Models<-data.frame(Models,Accuracy,Cross_Val_Accuracy,Yes_Correct_Clasified,No_Correct_Clasified,Yes_Correct_Clasified_CV,No_Correct_Clasified_CV)
#write.csv(Table_Models,"C:/Users/eduar/Documents/Table_Models.csv")

TN<-summary(train_file_clean$target)[2]#True Negative
TP<-summary(train_file_clean$target)[1]#True Positive
Table_Models$Accuracy<-(Yes_Correct_Clasified+No_Correct_Clasified)/(TN+TP)
Table_Models$Cross_Val_Accuracy<-(Yes_Correct_Clasified_CV+No_Correct_Clasified_CV)/(TN+TP)
Table_Models$Sensitivity<-(No_Correct_Clasified/TP)*100
Table_Models$Sensitivity_CV<-(No_Correct_Clasified_CV/TP)*100
Table_Models$Specificity<-(Yes_Correct_Clasified/TN)*100
Table_Models$Specificity_CV<-(Yes_Correct_Clasified_CV/TN)*100
Table_Models$Overfitting<-(Accuracy-Cross_Val_Accuracy)*100
Table_Models$Ensamble<-ifelse(Table_Models$Models=="ZeroR"|Table_Models$Models=="OneR",-1,ifelse(Table_Models$Models=="Bagging"|Table_Models$Models=="LogitBoost"|Table_Models$Models=="AdaBoostM1"|Table_Models$Models=="Stacking",1,0))
Table_Models<-Table_Models[order(Table_Models$Ensamble,Table_Models$Overfitting,Table_Models$Yes_Correct_Clasified_CV),]


library(ggplot2)
g<-ggplot(Table_Models, aes(Models,Sensitivity))
g+geom_bar(stat = "identity")
#==============Plotting the data====================
library(ggplot2)
library(reshape2)

# Everything on the same plot with smooth stat
ggplot(Table_Models, aes(Models,Sensitivity)) + 
  geom_point() + 
  stat_smooth() 
# Everything on the same plot normal data
ggplot(Table_Models, aes(Models,Specificity_CV)) +
  theme_bw() +
  geom_line()
# Individuals Graphs
ggplot(Table_Models, aes(Models,Overfitting)) +
  theme_bw() +
  geom_line() +
  facet_wrap(~ variable)

#===============IBk C & M Optimisation===================
WOW(IBk)
OF<-0
CVnew<-0
OFnew<-0
Cnew<-0
Knew<-0
Neighbour<-0
df_IBk <- data.frame(SAMPLE=double(),O_F=double(),OF_New=double(),K_New=double(),CV_New=double(),"OFnew+CVnew"=double(),T_R=double())
for (i in 1:20) {
  Neighbour=i
  IBk_Classifier<-IBk(train_file_clean$target~ ., data = train_file_clean,control=Weka_control(K=Neighbour))
  IBk_Classifier_Summary<-summary(IBk_Classifier)
  TR<-IBk_Classifier_Summary$details[[1]]
  
  #Cross Validation
  IBk_CV <- evaluate_Weka_classifier(IBk_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  #Cross Validation
  CV<-IBk_CV$details[[1]]
  OF<-TR-CV
  ifelse(-OFnew+CVnew>-OF+CV,OFnew,OFnew<-OF)
  ifelse(-OFnew+CVnew>-OF+CV,Knew,Knew<-Neighbour)
  ifelse(-OFnew+CVnew>-OF+CV,CVnew,CVnew<-CV)
  df_IBk[nrow(df_IBk) + 1,] = list(SAMPLE=i,O_F=OF,OF_New=OFnew,K_New=Knew,CV_New=CVnew,"OFnew+CVnew"=(CVnew-(OFnew*CVnew)),T_R=TR)
  
}
#===============J48 C & M Optimisation===================
WOW(J48)

OF<-0
CVnew<-0
OFnew<-0
Cnew<-0
Mnew<-0
for (i in 1:100) {
  confidence<-runif(1, 0, 0.25)
  minObjleaf<-floor(runif(1, 1,10)) 
  J48_Classifier<-J48(train_file_clean$target~ ., data = train_file_clean, control=Weka_control(C=confidence,M=minObjleaf))
  J48_Classifier_Summary<-summary(J48_Classifier)
  TR<-J48_Classifier_Summary$details[[1]]
  
  #Cross Validation
  J48_CV <- evaluate_Weka_classifier(J48_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  CV<-J48_CV$details[[1]]
  OF<-TR-CV
  ifelse(-OFnew+CVnew>-OF+CV,OFnew,OFnew<-OF)
  ifelse(-OFnew+CVnew>-OF+CV,Cnew,Cnew<-confidence)
  ifelse(-OFnew+CVnew>-OF+CV,Mnew,Mnew<-minObjleaf)
  ifelse(-OFnew+CVnew>-OF+CV,CVnew,CVnew<-CV)
  print(paste0(" OF ", format(round(OF,digits = 4), nsmall = 4)," OF New :", format(round(OFnew,digits = 4), nsmall = 4), " CNew :", 
               format(round(Cnew,digits = 4), nsmall = 4), " MNew :", format(round(Mnew,digits = 4), nsmall = 4),"  CV New ", 
               format(round(CV,digits = 4), nsmall = 4),"  OFnew+CVnew :", format(round(OFnew+CVnew,digits = 4), nsmall = 4),"  TR New :", format(round(TR,digits = 4), nsmall = 4)
  ) )
}
#=================Optimizing the attributes=================
GainRatioAttributeEval(train_file_clean$target~ . , data = train_file_clean)
train_file_clean_GainR<-subset(train_file_clean, select = c(-1,-2))
InfoGainAttributeEval(train_file_clean$target~ . , data = train_file_clean)
train_file_clean_InfoGain<-subset(train_file_clean, select = c(-1,-6,-17))

#==================Evaluate with new set InfoGain====================
OF<-0
CVnew<-0
OFnew<-0
Cnew<-0
Mnew<-0
for (i in 1:20) {
  Neighbour<-i+1
  
  IBk_Classifier<-IBk(train_file_clean_InfoGain$class~ ., data = train_file_clean_InfoGain,control=Weka_control(K=Neighbour))
  IBk_Classifier_Summary<-summary(IBk_Classifier)
  TR<-IBk_Classifier_Summary$details[[1]]
  
  #Cross Validation
  IBk_CV <- evaluate_Weka_classifier(IBk_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  #Cross Validation
  CV<-IBk_CV$details[[1]]
  OF<-TR-CV
  ifelse(-OFnew+CVnew>-OF+CV,OFnew,OFnew<-OF)
  ifelse(-OFnew+CVnew>-OF+CV,Knew,Knew<-Neighbour)
  ifelse(-OFnew+CVnew>-OF+CV,CVnew,CVnew<-CV)
  print(paste0(" OF ", format(round(OF,digits = 4), nsmall = 4)," OF New :", format(round(OFnew,digits = 4), nsmall = 4), " KNew :", 
               format(round(Knew,digits = 4), nsmall = 4), "  CV New ", format(round(CV,digits = 4), nsmall = 4),"  OFnew+CVnew :", format(round(OFnew+CVnew,digits = 4), nsmall = 4),"  TR New :", format(round(TR,digits = 4), nsmall = 4)
               ,"Value ",i) )
}

#==================Evaluate with new set InfoGain====================
OF<-0
CVnew<-0
OFnew<-0
Cnew<-0
Mnew<-0
for (i in 1:20) {
  Neighbour<-i+1
  
  IBk_Classifier<-IBk(train_file_clean_GainR$class~ ., data = train_file_clean_GainR,control=Weka_control(K=Neighbour))
  IBk_Classifier_Summary<-summary(IBk_Classifier)
  TR<-IBk_Classifier_Summary$details[[1]]
  
  #Cross Validation
  IBk_CV <- evaluate_Weka_classifier(IBk_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  #Cross Validation
  CV<-IBk_CV$details[[1]]
  OF<-TR-CV
  ifelse(-OFnew+CVnew>-OF+CV,OFnew,OFnew<-OF)
  ifelse(-OFnew+CVnew>-OF+CV,Knew,Knew<-Neighbour)
  ifelse(-OFnew+CVnew>-OF+CV,CVnew,CVnew<-CV)
  print(paste0(" OF ", format(round(OF,digits = 4), nsmall = 4)," OF New :", format(round(OFnew,digits = 4), nsmall = 4), " KNew :", 
               format(round(Knew,digits = 4), nsmall = 4), "  CV New ", format(round(CV,digits = 4), nsmall = 4),"  OFnew+CVnew :", format(round(OFnew+CVnew,digits = 4), nsmall = 4),"  TR New :", format(round(TR,digits = 4), nsmall = 4)
               ,"Value ",i) )
}

#======================Part 2 Numberic Prediction=====================
#===============  M5P===================
M5P_Classifier<-M5P(train_file_clean$age~ ., data = train_file_clean)
summary(M5P_Classifier)
#Cross Validation
M5P_CV <- evaluate_Weka_classifier(M5P_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
M5P_CV

#===============IBk===================
IBk_Classifier<-IBk(train_file_clean$age~ ., data = train_file_clean,control=Weka_control(K=1))
summary(IBk_Classifier)
#Cross Validation
IBk_CV <- evaluate_Weka_classifier(IBk_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
IBk_CV
WOW("M5P")

#===============  M5P improving ===================

df <- data.frame(SAMPLE=double(),O_F=double(),OF_New=double(),M_New=double(),CV_New=double(),"OFnew+CVnew"=double(),T_R=double())
OF<-0
CVnew<-0
OFnew<-0
Cnew<-0
Mnew<-0
TR<-0
MinInstances<-0
for (i in 1:50) {
  MinInstances<-i+1
  M5P_Classifier<-M5P(train_file_clean$age~ ., data = train_file_clean, control=Weka_control(M=MinInstances,U=FALSE))
  M5P_Classifier_Summary<-summary(M5P_Classifier)
  TR<-M5P_Classifier_Summary$details[[1]]
  
  #Cross Validation
  M5P_CV <- evaluate_Weka_classifier(M5P_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  #Cross Validation
  CV<-M5P_CV$details[[1]]
  OF<-TR-CV
  ifelse(CVnew-(OFnew*CVnew)>CV-(OF*CV),OFnew,OFnew<-OF)
  ifelse(CVnew-(OFnew*CVnew)>CV-(OF*CV),Mnew,Mnew<-MinInstances)
  ifelse(CVnew-(OFnew*CVnew)>CV-(OF*CV),CVnew,CVnew<-CV)
  df[nrow(df) + 1,] = list(SAMPLE=i,O_F=OF,OF_New=OFnew,M_New=Mnew,CV_New=CVnew,"OFnew+CVnew"=(CVnew-(OFnew*CVnew)),T_R=TR)
  
}

#==============Plotting the data====================
library(ggplot2)
library(reshape2)
df1 <-subset(df, select = c(-4))
df1 <- melt(df1, id.vars="SAMPLE")
# Everything on the same plot with smooth stat
ggplot(df1, aes(SAMPLE,value, col=variable)) + 
  geom_point() + 
  stat_smooth() 
# Everything on the same plot normal data
ggplot(df1, aes(SAMPLE,value, color = variable)) +
  theme_bw() +
  geom_line()
# Individuals Graphs
ggplot(df1, aes(SAMPLE,value)) +
  theme_bw() +
  geom_line() +
  facet_wrap(~ variable)
