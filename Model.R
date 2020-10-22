# toujours ouvrir le fichier .r dans la racine ou se trouve toutes les libraries
source("AFD_procedures.r")
trainData<-read.csv("ScoringTraining.csv",header = TRUE, sep = ",")[,2:12]
head(trainData)

require(FactoMineR)
library(Amelia)
library(rpart)
library(caTools)
library(lattice)
library(ggplot2)
library(caret)
library(sqldf)
library(MASS)
library(ROCR)

#########################
#Preparation des données#
#########################

#----- Préparation de données -----#

####### Question 1 #######

barplot(prop.table(table(trainData$SeriousDlqin2yrs)), 
        col = rainbow(2),
        ylim = c(0,1),
        main="Distribution des classes")

####### Question 2 #######

#par(mfrow = c(3,4))
boxplot(trainData$RevolvingUtilizationOfUnsecuredLines)
boxplot.stats(trainData$RevolvingUtilizationOfUnsecuredLines)


hist(trainData$RevolvingUtilizationOfUnsecuredLines, freq=FALSE, breaks="Sturges", col="darkgray")#bad
lines(density(trainData$RevolvingUtilizationOfUnsecuredLines), col="blue", lwd=2)
hist(trainData$age, freq=FALSE, breaks="Sturges", col="darkgray")#good
lines(density(trainData$age), col="blue", lwd=2)
hist(trainData$NumberOfTime30_59DaysPastDueNotWorse, freq=FALSE, breaks="Sturges", col="darkgray")#bad
hist(trainData$DebtRatio, freq=FALSE, breaks="Sturges", col="darkgray")#bad
hist(trainData$MonthlyIncome, freq=FALSE, breaks="Sturges", col="darkgray")#bad
hist(trainData$NumberOfOpenCreditLinesAndLoans, freq=FALSE, breaks="Sturges", col="darkgray")#good
lines(density(trainData$NumberOfOpenCreditLinesAndLoans), col="blue", lwd=2)
hist(trainData$NumberOfTimes90DaysLate, freq=FALSE, breaks="Sturges", col="darkgray")#bad
hist(trainData$NumberRealEstateLoansOrLines, freq=FALSE, breaks="Sturges", col="darkgray")#bad
hist(trainData$NumberOfTime60_89DaysPastDueNotWorse, freq=FALSE, breaks="Sturges", col="darkgray")#bad
hist(trainData$NumberOfDependents, freq=FALSE, breaks="Sturges", col="darkgray")#good

####### Question 3 #######

missmap(trainData)

####### Question 4 #######

cleanData=trainData
sapply(cleanData, function(x) sum(is.na(x)))
#which(is.na(cleanData$MonthlyIncome)) #Tells us the location of all NA values

cleanData$MonthlyIncome[which(is.na(cleanData$MonthlyIncome))] <- median(cleanData$MonthlyIncome, na.rm=TRUE) #Substitutes NA values for the median in that column
cleanData$NumberOfDependents[which(is.na(cleanData$NumberOfDependents))] <- median(cleanData$NumberOfDependents, na.rm=TRUE) #Substitutes NA values for the median in that column

missmap(cleanData,main="Clean data") #Only missing values in file variable
summary(cleanData)

#----- Equilibrage des données d'apprentissage -----#

####### Question 5 #######

set.seed(123)
cleanData = downSample(x=cleanData[, -ncol(cleanData)], y=factor(cleanData$SeriousDlqin2yrs))
prop.table(table(cleanData$SeriousDlqin2yrs))

split = sample.split(cleanData$SeriousDlqin2yrs, SplitRatio = 0.7)

TrainingData = subset(cleanData, split == TRUE)
TestData = subset(cleanData, split == TRUE)

#----- Identification des meilleurs prédicteurs parmi les variables -----#

####### Question 6 #######

par(mfrow = c(3,4))
boxplot(cleanData$SeriousDlqin2yrs, trainData$age,main="age")
boxplot(trainData$SeriousDlqin2yrs, trainData$RevolvingUtilizationOfUnsecuredLines,main="RevolvingUtilizationOfUnsecuredLines")
boxplot(trainData$SeriousDlqin2yrs, trainData$NumberOfTime30_59DaysPastDueNotWorse,main="NumberOfTime30_59DaysPastDueNotWorse",ylim=c(0,3))
boxplot(trainData$SeriousDlqin2yrs, trainData$DebtRatio,main="DebtRatio")
boxplot(trainData$SeriousDlqin2yrs, trainData$MonthlyIncome,main="MonthlyIncome")
boxplot(trainData$SeriousDlqin2yrs, trainData$NumberOfOpenCreditLinesAndLoans,main="NumberOfOpenCreditLinesAndLoans")
boxplot(trainData$SeriousDlqin2yrs, trainData$NumberOfTimes90DaysLate,main="NumberOfTimes90DaysLate")
boxplot(trainData$SeriousDlqin2yrs, trainData$NumberRealEstateLoansOrLines,main="NumberRealEstateLoansOrLines")
boxplot(trainData$SeriousDlqin2yrs, trainData$NumberOfTime60_89DaysPastDueNotWorse,main="NumberOfTime60_89DaysPastDueNotWorse")
boxplot(trainData$SeriousDlqin2yrs, trainData$NumberOfDependents,main="NumberOfDependents")

#########################
#Preparation des données#
#########################

####### Question 7 #######

DataChosen = sqldf("select age, DebtRatio, MonthlyIncome, NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate from TrainingData;")
ResAFD = AFD(DataChosen, TrainingData$SeriousDlqin2yrs)

plotAFD(ResAFD)

####### Question 8 #######

#-------- LDA --------#

data.lda = lda(TrainingData$SeriousDlqin2yrs ~ .,data=TrainingData[,c("age", "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate")])
data.lda$scaling                #facteur discriminant

PredictionLDA <- predict(data.lda)

head(PredictionLDA$x)                #variable discriminante (canonique)

tab = table(Predicted=PredictionLDA$class, TrainingData$SeriousDlqin2yrs)
tab

sensitivity(tab)              # Sensitivity
specificity(tab)              # Specificity

sum(diag(tab))/sum(tab)       # Accuracy = 61%

head(data.lda)
plot(data.lda)

#-------- QDA --------#

data.qda <- qda(TrainingData$SeriousDlqin2yrs ~.,data=TrainingData[,c("age", "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate")])

qda.values <- predict(data.qda, data=TrainingData)
tab = table(qda.values$class, TrainingData$SeriousDlqin2yrs)
tab

sensitivity(tab)              # Sensitivity
specificity(tab)              # Specificity

sum(diag(tab))/sum(tab)       # Accuracy = 62%

plot(qda.values$posterior[,2], qda.values$class, col=TrainingData$SeriousDlqin2yrs)

####### Question 9 #######

#-------- Régression logistique --------#

ResRL<- glm(TrainingData$SeriousDlqin2yrs ~ TrainingData$age+TrainingData$NumberOfTimes90DaysLate, data=TrainingData,family='binomial')

summary(ResRL)

PredictionRL<-predict(ResRL, TrainingData, type="response")

pred1 = ifelse(PredictionRL>0.5, 1, 0)
tab = table(Predicted = pred1, Actual = TrainingData$SeriousDlqin2yrs)
tab

sensitivity(tab)              # Sensitivity
specificity(tab)              # Specificity

sum(diag(tab))/sum(tab)       # Accuracy = 66%

#plot(ResRL)

#################################################
#Phase d'évaluation et règle de décision retenue#
#################################################

####### Question 11 #######

#-------- Courbe ROC & AUC --------#

table(TrainingData$SeriousDlqin2yrs,PredictionRL>0.5)

pred=prediction(PredictionRL,TrainingData$SeriousDlqin2yrs)
perf=performance(pred,"tpr", "fpr")
plot(perf,colorize = TRUE)

abline(a=0, b=1)

auc = performance(pred, "auc")
auc = unlist(slot(auc,"y.values"))
auc = round(auc, 4)

auc

legend(.6, .2, auc, title = "AUC")

#test

tableau = table(TrainingData$SeriousDlqin2yrs,TrainingData$NumberOfTimes90DaysLate)
tableau
barplot(tableau, beside = T,legend=T)
khi_test = chisq.test(tableau) 
khi_test 

attributes(khi_test)
