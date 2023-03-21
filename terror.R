setwd('F:/DATA MINING/PROGETTO')

a=read.csv('globalterrorism1.csv', sep=';', dec=',',na.strings = '')

table(a$nkill)
sapply(a, function(x)(sum(is.na(x))))


a$iskilled=ifelse(a$nkill>0,'yes','no')
table(a$iskilled)
prop.table(table(a$iskilled))

summary(a)

#togliamo motive,claimed,corp1,nperps,nperpcap,wea?detail, weapsubtype1,propextent,ransom,int_log,int_any,int_ideo
#missing oltre 20%
colnames(a)
a=a[,-c(15,19,22,23,24,26,29,28,32,34,36,37,39)]


b=na.omit(a)


#cerchiamo valori strani
library(funModeling)
library(dplyr)
status=df_status(b, print_results = F)
head(status%>% arrange(type))
head(status%>% arrange(unique))
head(status%>% arrange(-p_na))

library(caret)

numeric <- sapply(b, function(x) is.numeric(x)) 
numeric <-b[, numeric]
str(numeric)

b$iyear=as.factor(b$iyear)
b$imonth=as.factor(b$imonth)
b$iday=as.factor(b$iday)
b$country=as.factor(b$country)
b$specificity=as.factor(b$specificity)
b$doubtterr=as.factor(b$doubtterr)
b$multiple=as.factor(b$multiple)
b$success=as.factor(b$success)
b$suicide=as.factor(b$suicide)
b$guncertain1=as.factor(b$guncertain1)
b$individual=as.factor(b$individual)
b$property=as.factor(b$property)
b$ishostkid=as.factor(b$ishostkid)
b$INT_MISC=as.factor(b$INT_MISC)
b$iskilled=as.factor(b$iskilled)

numeric <- sapply(b, function(x) is.numeric(x)) 
numeric <-b[, numeric]
str(numeric)

#nzv
nzv = nearZeroVar(b, saveMetrics = TRUE)
nzv

#leviamo ishostkid,individual,suicide 
colnames(b)
b=b[,-c(11,19,24)]

#provare un set.seed
c=b[sample(nrow(b),8000),]

library(e1071)
scaled_bc <- preProcess(c, method = c("scale", "BoxCox"))
scaled_bc
scaled_bc$bc

#aggiunge var scalate al dataset
all_bc=predict(scaled_bc, newdata = c)
head(all_bc)
head(c)

#effetti della trasformazione
par(mfrow=c(1,2))
hist(c$nwound)
hist(all_bc$nwound)
par(mfrow=c(1,1))

#togliamo city,target1,gname poich? hanno troppi livelli
colnames(all_bc)
all_bc=all_bc[,-c(6,14,16)]
#togliamo year,country,targetsubt,nationality per troppi liv
colnames(all_bc)
all_bc=all_bc[,-c(1,4,12,13)]


#creiamo valid(test) e train
set.seed(1234)
split <- createDataPartition(y=c$iskilled, p = 0.66, list = FALSE)
train <- all_bc[split,]
test <- all_bc[-split,]



#CLASSIFICATION TREE
?expand.grid
set.seed(1)
ctrl_tree <- trainControl(method = "cv", number=10 , savePredictions=T,search="grid", summaryFunction = twoClassSummary , classProbs = TRUE)
tree <- train(iskilled~., data=train, method="rpart",  
            trControl=ctrl_tree, tuneLength=10)

tree
getTrainPerf(tree)

pred_tree=predict(tree,newdata=test)
confusionMatrix(pred_tree, test$iskilled)

# Accuracy : 0.8062 Kappa : 0.6109  






table(all_bc$iskilled)/nrow(all_bc)

table(train$iskilled)/nrow(train)
table(test$iskilled)/nrow(test)

library(rpart)
library(rpart.plot)

set.seed(1)  
default.ct <- rpart(iskilled ~ ., data = train, method = "class")
# plot tree

rpart.plot(default.ct, type = 4, extra = 1)
rpart.plot(default.ct, type = 4, extra = 101,  split.font = 0.9, ycompress=FALSE, cex=.7)

# cp=penaliazion for large tree 
# cp=0 no penalization, largest tree
# cp high smaller tree
#### FIND maximum tree, drop default best pruning cp by by rpart
set.seed(1)
deeper.ct <- rpart(iskilled ~ ., data = train, method = "class", cp = 0, minsplit = 1)
rpart.plot(deeper.ct, type = 4, extra = 1)


#step dell'albero
a=data.frame(deeper.ct$frame$var)
table(a)

#importanza variabili
vi=data.frame(deeper.ct$variable.importance)
vi

# rel_error= absolute misclassified rate in training set / 
# absolute misclassification error in the tree with 0 split (relative)
# tree with 0 split= root node: no model, no tree, heterogeneity of target in the data

# xerror = relative misclassification rate using 10fold cv

table(train$iskilled)/nrow(train)
0.467*0.472*100
0.467*0.003*100

# here you can see the overfitting: 
# rel error decrease at zero, xerror do not, but increases from a certain dimensionality

#scegliamo la complessit? migliore
set.seed(1)
cv.ct <- rpart(iskilled ~ ., data = train, method = "class", 
               cp = 0.00001, minsplit = 5, xval = 5)
printcp(cv.ct)


# 1) let's prune cv.ct by cp with lower xerror cp=0.0034364
cv.ct$cptable[,"xerror"]

set.seed(1)
pruned1 <- prune(cv.ct, 
                 cp = cv.ct$cptable[which.min(cv.ct$cptable[,"xerror"]),"CP"])

length(pruned1$frame$var[pruned1$frame$var == "<leaf>"])
#tree migliore
rpart.plot(pruned1, type = 4, extra = 1)

#nsplit=5, rel error=0.408, xerror=0.412

best_pruned.ct=pruned1
#non proviamo lo split minimizzando rel err-xerr poich? darebbe nsplit=2 e non ha senso

# fit our best_pruned.ct on test data and see confusion matrix

library(caret)
best_tree.pred <- predict(best_pruned.ct, test,type = "class")
confusionMatrix(best_tree.pred, test$iskilled, positive="yes")
#risulta un'accuracy di 0.8076 e una Kappa di 0.613
#non bellissimo



#set.seed(1)
#Ctrl <- trainControl(method = "cv" , number=5, classProbs = TRUE) # this gives ROC as measure, without this is accuracy
#rpartTune <- train(iskilled ~ ., data = train, method = "rpart", 
#                   tuneLength = 15, trControl = Ctrl,minsplit = 5)

##best tuned
#rpartTune
#Vimportance <- varImp(rpartTune)
#plot(Vimportance)




# performance on test set
testpred <- predict(rpartTune, test,type = "raw")
confusionMatrix(testpred, test$iskilled, positive="yes")
#risulta un'accuracy di 0.8168 e una Kappa di 0.6321


#proviamo a massimizzare la sensitivity in modo da controllare gli attacchi con veri morti
set.seed(7)
metric <- "Sens"   # Kappa , AUC , Sens.....etc
control <- trainControl(method="cv", number=10, search="grid", summaryFunction = twoClassSummary, classProbs = TRUE)
tunegrid <- expand.grid(.mtry=c(1:5))
fit2 <- train(iskilled~., data=all_bc, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
#non parte il comando perch? fa random forest
fit2


# best tuned mtry with highest Sens
plot(fit2)

# performance by mtry
fit2$results

# final model and performance
getTrainPerf(fit2)

# metrics for each fold with best mtry
fit2$resample

# final model and performance....which is the mean of 10 folds
mean(fit2$resample$ROC)
mean(fit2$resample$Spens)

#togliamo le var anche da c
colnames(c)
colnames(all_bc)
c=c[,-c(1,4,6,13,14,15,16)]

#NAIVE BAYES

#provo con train di caret


set.seed(1)
ctrl_nb <- trainControl(method = "cv", number=10 , savePredictions=T,search="grid", summaryFunction = twoClassSummary , classProbs = TRUE)
#tunegrid_knn <- expand.grid(k=c(15, 30, 45))
nb <- train(iskilled~., data=train, method="nb",  
              trControl=ctrl_nb, tuneLength=10)

nb
getTrainPerf(nb)

pred_nb=predict(nb,newdata=test)
confusionMatrix(pred_nb, test$iskilled)

#Accuracy : 0.7698 Kappa : 0.5382  








library(klaR)
#creiamo il dataset con le sole covariate senza target
predictors=all_bc[,-17]
str(predictors)

naive_all1 <- NaiveBayes(iskilled ~ ., data = c, usekernel = FALSE)

ls(naive_all1$tables)


# P(y) a priori
naive_all1$apriori
# P(x/class j) =conditional distribution x/cj in rows, sum of each row = 1
naive_all1$tables$attacktype1_txt
#nel caso ci siano morti per il 38.7% sono dovuti a armed assault 
#e per il 37.4% a bombing/explosion

# continuous X
# P(x/class j) =conditional densities x/cj: Mean (I col) and stand dev ((II col)) of gaussian dentities  in rows
naive_all1$tables$nwound
#quando c'? almeno un morto, la media di feriti ? 5,
#quando non ci sono morti la media ? 1
##la prima colonna sono le medie, la seconda colonna le deviazioni standard

par(mfrow=c(1,2))
hist(c$nwound[which(c$iskilled=='no')], main='target no')
hist(c$nwound[which(c$iskilled=='yes')], main='target yes')

# see if there is possible 0 problem .
naive_all1$tables 
naive_all1$tables$guncertainyes
table(c$iskilled,c$guncertainyes)

#applichiamo correzione di Laplace
naive_all4 <- NaiveBayes(iskilled ~ ., data = c, laplace = 100)

#va aggiungere un valore 100 alle sei celle precedenti

table(c$iskilled,c$guncertainyes)
table(c$iskilled,c$guncertainyes)+100


# objects contained in predict of a nb model (Klar)
pred <- predict(naive_all4, c, type="class")
ls(pred)
# predicted y
head(pred$class)
# predicted probs
head(pred$posterior)


# confusion matrix on train
table(pred=pred$class, true=c$iskilled)/nrow(c)

#...divide data in train and test
# do nb model on train and evaluate it on validation.....

set.seed(1234)
split <- createDataPartition(y=c$iskilled, p = 0.66, list = FALSE)
train <- c[split,]
test <-  c[-split,]

naive_all4 <- NaiveBayes(iskilled ~ ., data = train, laplace = 100)

pred <- predict(naive_all4, train, type="class")
ls(pred)
# predicted y
head(pred$class)
# predicted probs
head(pred$posterior)


# confusion matrix on test
pred_test <- predict(naive_all4, test, type="class")

table(pred_test$class, true=test$iskilled)

# confusion matrix on test using caret..more statistics
library(caret)
confusionMatrix(pred_test$class, test$iskilled)

#accuracy di 0.7823, kappa di 0.5597




#NEURAL NETWORKS


#provo train con caret

set.seed(1)
ctrl_nn <- trainControl(method = "cv", number=10 , savePredictions=T,search="grid", summaryFunction = twoClassSummary , classProbs = TRUE)
#tunegrid_knn <- expand.grid(k=c(15, 30, 45))
nn <- train(iskilled~., data=train, method="nnet",  
              trControl=ctrl_nn, tuneLength=10)


nn
getTrainPerf(nn)

pred_nn=predict(nn,newdata=test)
confusionMatrix(pred_nn, test$iskilled)

# Accuracy : 0.8246  Kappa : 0.6478  



























library(dplyr)

sd=as_tibble(all_bc)
sd

library(funModeling)
status=df_status(sd, print_results = F)
status

nzv = nearZeroVar(sd, saveMetrics = TRUE)
nzv
# no factor covariates, no missing, no znv: ok
#(magari togliere qualche var factor)

# boxplot
featurePlot(x = sd[, 13], 
            y = sd$iskilled, 
            plot = "box", 
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),  
            layout = c(1,1), 
            auto.key = list(columns = 3))


# density plot better visualization
qplot(nwound, color=iskilled, data=sd, geom='density')
qplot(WidthCh1, color=Class, data=sd, geom='density')


set.seed(1234)
split <- createDataPartition(y=all_bc$iskilled, p = 0.66, list = FALSE)
train <- all_bc[split,]
test <-  all_bc[-split,]

#model selection usando tree 
set.seed(1)
cvCtrl <- trainControl(method = "cv", number=10, search="grid", classProbs = TRUE)

rpartTuneCvA <- train(iskilled ~ ., data = train, method = "rpart",
                      tuneLength = 10,
                      trControl = cvCtrl)

# best accuracy using best cp
rpartTuneCvA

# final model
getTrainPerf(rpartTuneCvA)
plot(rpartTuneCvA)

#aumentando la complessit? peggiora

# var imp of the tree
varImp(object=rpartTuneCvA)
plot(varImp(object=rpartTuneCvA),main="train tuned - Variable Importance")

# complete performance by cp 
rpartTuneCvA$results
# each row is the mean of metrics of 10 folds, by cp)
#otteniamo un'accuracy di 0.796 e un Kappa di 0.59

vi=as.data.frame(rpartTuneCvA$finalModel$variable.importance)
head(vi)

viname=row.names(vi)
head(viname)

training2=train[,viname]




library(Boruta)
try_boruta=sd

set.seed(123)
boruta.train <- Boruta(iskilled~., data = try_boruta, doTrace = 1)
print(boruta.train)

plot(boruta.train, xlab = "features", xaxt = "n", ylab="MDI")
#' Blue boxplots correspond to minimal, average and maximum Z score of MDI of an attribute. 
#' Red un-important feature 
#' yellow tentative/at limit important feature 
#' green important feature


# select vars: drop tantative and unimportant vars
final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)
# boruta metrics
boruta.df <- attStats(final.boruta)
boruta.df


#selezioniamo var pi? importanti
selected=getSelectedAttributes(final.boruta, withTentative = F)

# select data from selected vars
boruta_selected=try_boruta[,selected]
head(boruta_selected)


colnames(boruta_selected)
colnames(all_bc)

#usiamo boruta, rf troppo pesante computazionalmente, togliamo iday e guncertaintype dal dataset

colnames(train)
train=train[,-c(2,10)]
test=test[,-c(2,10)]


# nnet command dont want string y
y=ifelse(train$iskilled=="yes",1,0)
str(y)

train$doubtterr<-recode(train$doubtterr, recodes="'no'=0; else=1")
train$multiple<-recode(train$multiple, recodes="'no'=0; else=1")
train$success<-recode(train$success, recodes="'no'=0; else=1")
train$property<-recode(train$property, recodes="'no'=0; else=1")
train$INT_MISC<-recode(train$INT_MISC, recodes="'no'=0; else=1")


#provo a dummizzare
dummies <- dummyVars(imonth ~ ., data = train , fullRank = T, na.action = na.pass)
dummized = data.frame(predict(dummies, newdata = train))
head(dummized)
str(dummized)


all_train=cbind(dummized, y)

all_train$iskilled.yes=NULL

install.packages("nnet")
library(nnet)
# do a MPL with classical tuning parm
set.seed(7)
mynet <- nnet(all_train[,-114], y ,  entropy=T, size=3, decay=0.1, maxit=2000, trace=T)

mynet

# see architecture
library(NeuralNetTools)
plotnet(mynet, alpha=0.6)

mynet.pred <- as.numeric(predict(mynet, all_train[,-114], type='class'))
table(mynet.pred,y)


#per la confusion matrix devono essere factor
mynet.pred=as.factor(mynet.pred)
y=as.factor(y)
confusionMatrix(mynet.pred,y)

#otteniamo un'accuracy del 0.8485 e Kappa 0.6965

#proviamo a migliorare la Sensitivity
set.seed(7)
metric_nnet <- "accuracy"
ctrl_nnet = trainControl(method="cv", number=10, search = "grid")
nnetFit_def <- train(iskilled~., data=train,
                     method = "nnet",
                     preProcess = "range", 
                     trControl=ctrl_nnet,
                     trace = TRUE, # use true to see convergence
                     maxit = 100)
getTrainPerf(nnetFit_def)
confusionMatrix(nnetFit_def)
print(nnetFit_def)
plot(nnetFit_def)

#copio train e test per lavorarci su nnet
train_nnet=train
test_nnet=test

#matrice di confusione sul test set usando come metrica l'accuracy
test_nnet$pred1 = predict(nnetFit_def, test_nnet, "raw")
confusionMatrix(test_nnet$pred1, test_nnet$iskilled)
#accuracy=0.8139 kappa=0.6264





#K NEIGHREST NEIGHBOR

set.seed(1)
ctrl_knn <- trainControl(method = "cv", number=10 , savePredictions=T,search="grid", summaryFunction = twoClassSummary , classProbs = TRUE)
tunegrid_knn <- expand.grid(k=c(15, 30, 45))
knn <- train(iskilled~., data=train, method="knn",  
             tuneGrid=tunegrid_knn, trControl=ctrl_knn, tuneLength=10)


knn
getTrainPerf(knn)

pred=predict(knn,newdata=test)
confusionMatrix(pred, test$iskilled)
#otteniamo un'accuracy di 0.7878 e un kappa di 0.5761



#GLM
set.seed(1)
ctrl_glm <- trainControl(method = "cv" , number=10, summaryFunction = twoClassSummary , classProbs = TRUE) # this gives ROC as measure, without this is accuracy
glm <- train(iskilled ~ ., data = train, method = "glm", 
             trControl=ctrl_glm)

getTrainPerf(glm)

predglm=predict(glm,newdata=test)
confusionMatrix(predglm, test$iskilled)

#accuracy di 0.8227 e kappa di 0.6441 


#LOGISTIC 
library(MASS)
fit <- glm(iskilled~. , data=train, family="binomial")
#summary(fit)
step <- stepAIC(fit, direction="both")


 
#LASSO

# lastly fit a lasso model
set.seed(1234)
grid_lasso = expand.grid(.alpha=1,.lambda=seq(0, 1, by = 0.01))
Control_lasso=trainControl(method= "cv",number=10, classProbs=TRUE)
glm_lasso=train(iskilled~.,data=train , method = "glmnet", family ="binomial",
                trControl = Control_lasso, tuneLength=5, tuneGrid=grid_lasso)
glm_lasso
plot(glm_lasso)

ls(glm_lasso)
coef(glm_lasso$finalModel, s=glm_lasso$bestTune$lambda)

confusionMatrix(glm_lasso)

#si ottiene solo l'accuracy che ? pari a 0.8175



#RANDOM FOREST
set.seed(1)
cvCtrl_rf <- trainControl(method = "cv", number=10, search="grid", classProbs = TRUE,
                       summaryFunction = twoClassSummary)
rfTune_rf <- train(iskilled ~ ., data = train, method = "rf",
                tuneLength = 10,
                trControl = cvCtrl_rf)
rfTune_rf

getTrainPerf(rfTune_rf)
#ROC di 0.91!! 

pred_rf=predict(rfTune_rf,newdata=test)
confusionMatrix(pred_rf, test$iskilled)
# Accuracy : 0.8349   Kappa : 0.6685   


#vimp=varImp(rfTune)
#plot(varImp(object=rfTune),main="train tuned - Variable Importance")
#vimp=data.frame(vimp[1])
#vimp$var=row.names(vimp)
#head(vimp)




#tree
#naive bayes
#neural net
#lasso
#glm
#logistica
#knn
#rf?

#STEP 2
#comparing models 
?resamples
results <- resamples(list(glm_PreProc=glm, knnPP=knn,
                          nnetPP=nn ,nbPP=nb, rfPP=rfTune_rf, treePP=tree))

bwplot(results)


# estimate probs P(M)
test$p1 = predict(glm       , test, "prob")[,1]
test$p2 = predict(knn         , test, "prob")[,1]
test$p3 = predict(nn    , test, "prob")[,1]
test$p4 = predict(nb     , test, "prob")[,1]
test$p5 = predict(rfTune_rf, test, "prob")[,1]
test$p6=  predict(tree, test, "prob")[,1]


library(pROC)
# roc values
r1=roc(iskilled ~ p1, data = test)
r2=roc(iskilled ~ p2, data = test)
r3=roc(iskilled ~ p3, data = test)
r4=roc(iskilled ~ p4, data = test)
r5=roc(iskilled ~ p5, data = test)
r6=roc(iskilled ~ p6, data = test)


plot(r1)
plot(r2,add=T,col="red")
plot(r3,add=T,col="blue")
plot(r4,add=T,col="yellow")
plot(r5,add=T,col="violet")
plot(r6,add=T,col="green")
#la curva migliore, poich? sopra a tutte le altre, risulta quella della rf


auc(r1)
auc(r2)
auc(r3)
auc(r4)
auc(r5)
auc(r6)
#l'auc maggiore risulta quello della rf pari a 0.9153


#STEP3

# take probabilities fitted on the validation set 
predP <- predict(rfTune_rf, test ,type = "prob")

df=data.frame(cbind(test$iskilled , predP))
head(df)
colnames(df)=c("iskilled","ProbNo","ProbYes")
head(df)

df=df[,c(1,3)]
head(df)
tail(df)

# create a cycle: for each ProbM create counts of confusion matrices (TP, FN....) and rates (TPR, FPR, )
library(dplyr)
# for each threshold, find tp, tn, fp, fn and the sens=prop_true_M, spec=prop_true_R, precision=tp/(tp+fp)



thresholds <- seq(from = 0, to = 1, by = 0.01)
prop_table <- data.frame(threshold = thresholds, prop_true_yes = NA,  prop_true_no = NA, true_yes = NA,  true_no = NA ,fn_yes=NA)

for (threshold in thresholds) {
  pred <- ifelse(df$ProbYes > threshold, "yes", "no")
  pred_t <- ifelse(pred == df$iskilled, TRUE, FALSE)
  
  group <- data.frame(df, "pred" = pred_t) %>%
    group_by(iskilled, pred) %>%
    dplyr::summarise(n = n())
  
  group_yes <- filter(group, iskilled == "yes")
  
  true_yes=sum(filter(group_yes, pred == TRUE)$n)
  prop_yes <- sum(filter(group_yes, pred == TRUE)$n) / sum(group_yes$n)
  
  prop_table[prop_table$threshold == threshold, "prop_true_yes"] <- prop_yes
  prop_table[prop_table$threshold == threshold, "true_yes"] <- true_yes
  
  fn_yes=sum(filter(group_yes, pred == FALSE)$n)
  # true Yes predicted as No
  prop_table[prop_table$threshold == threshold, "fn_yes"] <- fn_yes
  
  
  group_no <- filter(group, iskilled == "no")
  
  true_no=sum(filter(group_no, pred == TRUE)$n)
  prop_no <- sum(filter(group_no, pred == TRUE)$n) / sum(group_no$n)
  
  prop_table[prop_table$threshold == threshold, "prop_true_no"] <- prop_no
  prop_table[prop_table$threshold == threshold, "true_no"] <- true_no
  
}

# changing this the program can be re-used
# to replicate this script: dataset=df,   target=Class, target modalities, M, R

head(prop_table, n=10)
tail(prop_table, n=10)



# add other missing measures

# n of validation set    
prop_table$n=nrow(test)

# fp by difference of   n and            tn,                 tp,         fn, 
prop_table$fp_yes=nrow(test)-prop_table$true_no-prop_table$true_yes-prop_table$fn_yes

# find accuracy
prop_table$acc=(prop_table$true_no+prop_table$true_yes)/nrow(test)

# find precision
prop_table$prec_yes=prop_table$true_yes/(prop_table$true_yes+prop_table$fp_yes)

# verify not having NA metrics at start or end of data 
tail(prop_table)
# we have typically some NA in the precisiona t the boundary..put 0


library(Hmisc)
#impute NA as 0, this occurs typically for precision
prop_table$prec_yes=impute(prop_table$prec_yes, 1)
colnames(prop_table)

# drop counts, PLOT only metrics
# drop counts, PLOT only metrics
prop_table2 = prop_table[,-c(4:8)] 
head(prop_table2)


# plot measures vs soglia
# before we must impile data vertically: one block for each measure
library(dplyr)
library(tidyr)

gathered=prop_table2 %>%
  gather(x, y, prop_true_yes:prec_yes)

# plot measures 
library(ggplot2)
gathered %>%
  ggplot(aes(x = threshold, y = y, color = x)) +
  geom_point() +
  geom_line() +
  scale_color_brewer(palette = "Set1") +
  labs(y = "measures",
       color = "M: event\nR: nonevent")

# zoom
gathered %>%
  ggplot(aes(x = threshold, y = y, color = x)) +
  geom_point() +
  geom_line() +
  scale_color_brewer(palette = "Set1") +
  labs(y = "measures",
       color = "M: event\n R: nonevent") +
  coord_cartesian(xlim = c(0.4, 0.7))


#plot accuracy per scegliere soglia
gathered_acc=prop_table2 %>%
  gather(x, y,acc)

library(ggplot2)
gathered_acc %>%
  ggplot(aes(x = threshold, y = y, color = x)) +
  geom_point() +
  geom_line() +
  scale_color_brewer(palette = "Set1") +
  labs(y = "measures",
       color = "M: event\nR: nonevent")


#plot precision
gathered_prec=prop_table2 %>%
  gather(x, y,prec_yes)

library(ggplot2)
gathered_prec %>%
  ggplot(aes(x = threshold, y = y, color = x)) +
  geom_point() +
  geom_line() +
  scale_color_brewer(palette = "Set1") +
  labs(y = "measures",
       color = "M: event\nR: nonevent")

#plot sensitivity
gathered_sens=prop_table2 %>%
  gather(x, y,prop_true_yes)

library(ggplot2)
gathered_sens %>%
  ggplot(aes(x = threshold, y = y, color = x)) +
  geom_point() +
  geom_line() +
  scale_color_brewer(palette = "Set1") +
  labs(y = "measures",
       color = "M: event\nR: nonevent")


#plot specificity
gathered_spec=prop_table2 %>%
  gather(x, y,prop_true_no)

library(ggplot2)
gathered_spec %>%
  ggplot(aes(x = threshold, y = y, color = x)) +
  geom_point() +
  geom_line() +
  scale_color_brewer(palette = "Set1") +
  labs(y = "measures",
       color = "M: event\nR: nonevent")



#guardando il plot con tutte le misure, decidiamo di scegliere come soglia 0.51



#step4
# newdata=scoredata simulato

score=test[c(1:800),-c(15:21)]
score$prob = predict(rfTune_rf, score, "prob")
head(score$prob)
probyes=score$prob[,2]
head(probyes)
score$pred_y=ifelse(probyes>0.51, "yes","no")
head(score)


