######################################################################
# Important Note:
# This code is written using R v 4.0 on Mac
# The code can take upto 2 hours to run due to the time taken by the model training steps.
# 
######################################################################


# Install the required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(stringr)) install.packages("stringr")
if(!require(readr)) install.packages("readr")
if(!require(tidyr)) install.packages("tidyr")
if(!require(lubridate)) install.packages("lubridate")
if(!require(tidyr)) install.packages("tidyr")
if(!require(knitr)) install.packages("knitr")
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(matrixStats)) install.packages("matrixStats")
if(!require(rpart)) install.packages("rpart")


# load libraries
library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(ggplot2)
library(stringr)
library(readr)
library(tidyr)
library(lubridate)
library(knitr)
library(matrixStats)
library(rpart)


############################################################################################################
# Data download from github. 
# The files have been stored as rda files split due to file size restriction
############################################################################################################

# Load the data which is split and stored as 10 rda files on git hub
for(i in 1:10){
  dl <- tempfile()
  download.file(paste("https://github.com/rohitgyl/Capstone_BankData/raw/main/train_data",i,".rda",sep=""), dl)
  load(dl)
  }

# combine the  data
full_data_set <- rbind(train_data1,train_data2,train_data3,train_data4,
                    train_data5,train_data6,train_data7,train_data8,train_data9,train_data10)

# remove sub data sets from memory
rm(train_data1,train_data2,train_data3,train_data4,
      train_data5,train_data6,train_data7,train_data8,train_data9,train_data10)


# get target variable which is second col in file
y <- full_data_set[,2]

# get remaining variables on which target depends
x <- full_data_set[,3:202]

# convert to matrix 
x_m <- as.matrix(x)

# convert target var from 0/1 to no/yes
y_factor <- y%>% mutate(target =ifelse(target=="1","yes","no")) %>% mutate(target = factor(target))


##########################################################
# Create edx set (which will be used for model building)
# validation set (final hold-out test set)
# 
##########################################################

set.seed(1, sample.kind = "Rounding")    # if using R 3.6 or later

test_index <- createDataPartition(y = y_factor$target, times = 1, p = 0.2, list = FALSE)
edx_y <- y_factor[-test_index,]
edx_x <- x_m[-test_index,]

validation_y <- y_factor[test_index,]
validation_x <- x_m[test_index,]

# remove unwanted variables from memory
rm(y_factor,x_m,x,y,full_data_set)


############################################################################################################
# Data preparation and transfomration
############################################################################################################


# get column std deviations 
sds <- colSds(edx_x)

# check if any variables can be dropped due to near zero variance--> no such cases found
nzv <- nearZeroVar(edx_x)

# print nzv length
length(nzv)%>% kable(col.names = "Variables with near zero var","simple")

# check lowest 5 
as.data.frame(sds) %>% arrange(sds) %>% slice (1:5) %>% kable(col.names = "Lowest 5 Std Devs","simple")


# standardize the variables 
x_centered <- sweep(edx_x, 2, colMeans(edx_x))
x_scaled <- sweep(x_centered, 2, colSds(edx_x), FUN = "/")

rm(x_centered)


##########################################################
# Data exploration 
##########################################################

# Check sample data to inspect data attributes
head(cbind(edx_y,x_scaled[,1:3],x_scaled[,199:200]) ) %>% knitr::kable("simple")

# dimension of dataset
dim(cbind(edx_y,x_scaled)) %>% knitr::kable(col.names = "Dimensions: rows, columns","simple")


########################################################################################
# check proportion of yes/no cases -- 10% are yes cases i.e class imbalance is present
# i.e. overall accuracy not enough - check balanced score and sensitivity
########################################################################################
table(edx_y) %>% knitr::kable(col.names = c("Class","Count"),"simple")

#########################################################################
# perform principal component analysis to see we can keep transform the 200 variables and work with a smaller set
# for our analysis without loosing much information
#########################################################################

pca <- prcomp(x_scaled)
pca_summary <- summary(pca) 

importance <- as.matrix(pca_summary$importance)

# review cumulative importance of pca components. conclude to keep all variables
importance[3,]

rm(pca)


########################################################################################
# check trends of each variable for target yes/no by creating box plots
# as we have 200 variables, will split into 5 graphs with 40 per graph
########################################################################################

# combine x and y for plotting
tmp <- cbind(x_scaled,edx_y) 

# graph for var_0-var_40
tmp %>% select (var_0,var_1,var_2,var_3,var_4,var_5,var_6,var_7,var_8,var_9,var_10,
                var_11,var_12,var_13,var_14,var_15,var_16,var_17,var_18,var_19,var_20,
                var_21,var_22,var_23,var_24,var_25,var_26,var_27,var_28,var_29,var_30,
                var_31,var_32,var_33,var_34,var_35,var_36,var_37,var_38,var_39,var_40,
                target)%>%
  gather(key = "var", value = "value", -target) %>%
  ggplot(aes(var,value,fill=target))+
  geom_boxplot()+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  ggtitle("Box plot by target for Var0-Var40")


# graph for var_41-var_80
tmp %>% select (
                var_41,var_42,var_43,var_44,var_45,var_46,var_47,var_48,var_49,var_50,
                var_51,var_52,var_53,var_54,var_55,var_56,var_57,var_58,var_59,var_60,
                 var_61,var_62,var_63,var_64,var_65,var_66,var_67,var_68,var_69,var_70,
                 var_71,var_72,var_73,var_74,var_75,var_76,var_77,var_78,var_79,var_80,
                target)%>%
  gather(key = "var", value = "value", -target) %>%
  ggplot(aes(var,value,fill=target))+
  geom_boxplot()+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  ggtitle("Box plot by target for Var41-Var80")


# graph for var_81-var_120
tmp %>% select (
    var_81,var_82,var_83,var_84,var_85,var_86,var_87,var_88,var_89,var_90,
  var_91,var_92,var_93,var_94,var_95,var_96,var_97,var_98,var_99,var_100,
  var_101,var_102,var_103,var_104,var_105,var_106,var_107,var_108,var_109,var_110,
   var_111,var_112,var_113,var_114,var_115,var_116,var_117,var_118,var_119,var_120,
  target)%>%
  gather(key = "var", value = "value", -target) %>%
  ggplot(aes(var,value,fill=target))+
  geom_boxplot()+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  ggtitle("Box plot by target for Va81-Var120")

# graph for var_121-var_160
tmp %>% select (
   var_121,var_122,var_123,var_124,var_125,var_126,var_127,var_128,var_129,var_130,
   var_131,var_132,var_133,var_134,var_135,var_136,var_137,var_138,var_139,var_140,
   var_141,var_142,var_143,var_144,var_145,var_146,var_147,var_148,var_149,var_150,
   var_151,var_152,var_153,var_154,var_155,var_156,var_157,var_158,var_159,var_160,
  target)%>%
  gather(key = "var", value = "value", -target) %>%
  ggplot(aes(var,value,fill=target))+
  geom_boxplot()+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  ggtitle("Box plot by target for Var121-Var160")


# graph for var_161-var_199
tmp %>% select (
   var_161,var_162,var_163,var_164,var_165,var_166,var_167,var_168,var_169,var_170,
   var_171,var_172,var_173,var_174,var_175,var_176,var_177,var_178,var_179,var_180,
  var_181,var_182,var_183,var_184,var_185,var_186,var_187,var_188,var_189,var_190,
  var_191,var_192,var_193,var_194,var_195,var_196,var_197,var_198,var_199,
  target)%>%
  gather(key = "var", value = "value", -target) %>%
  ggplot(aes(var,value,fill=target))+
  geom_boxplot()+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  ggtitle("Box plot by target for Var161-Var199")


####################################################################################
# Modeling 
# rpart, glm, lda with cross validation
####################################################################################


# set seed for consistent results
set.seed(1, sample.kind = "Rounding")    # if using R 3.6 or later

# split edx data set further into 20% for test and 80% for training
test_index <- createDataPartition(edx_y$target, times = 1, p = 0.2, list = FALSE)
test_x <- x_scaled[test_index,]
test_y <- edx_y[test_index,]
train_x <- x_scaled[-test_index,]
train_y <- edx_y[-test_index,]


# set train control for cross validation
ctrl <- trainControl(method="cv", number = 10)


####################################################################################
# Model 1 - Rpart
####################################################################################

# set seed for consistent results
set.seed(1, sample.kind = "Rounding") 


# train using rpart
train_rpart <- train(train_x, train_y$target, 
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.05, len = 5)),
                     trControl = ctrl)

# plot to check if optimal cp found
ggplot(train_rpart)

# get predictions on test set
rpart_preds <- predict(train_rpart, as.data.frame(test_x))

# rpart accuracy,  sensitivity, precision
rpart_acc <- confusionMatrix(rpart_preds, test_y$target, positive ="yes")$overall["Accuracy"]  
rpart_sens <- sensitivity(rpart_preds, test_y$target, positive ="yes")
rpart_prec <- precision(rpart_preds, test_y$target, relevant ="yes")

# Add rpart results to the table

results <- data_frame(method = "Rpart", Accuracy =rpart_acc , Sensitivity=rpart_sens, Precision = rpart_prec)

results%>%  knitr::kable(col.names = c("Model","Accuracy","Sensitivity","Precision"),"simple")

####################################################################################
# Model 2 - glm
####################################################################################
# set seed for consistent results
set.seed(1, sample.kind = "Rounding") 

# train using glm
train_glm <- train(train_x, train_y$target, method = "glm",
                   trControl = ctrl)

# get predictions on test set
glm_preds <- predict(train_glm, test_x)

# glm accuracy,  sensitivity, precision
glm_acc <- confusionMatrix(glm_preds, test_y$target, positive ="yes")$overall["Accuracy"]  
glm_sens <- sensitivity(glm_preds, test_y$target, positive ="yes")
glm_prec <- precision(glm_preds, test_y$target, relevant ="yes")

# Add glm results to the table
results <- bind_rows(results,
                                          data_frame(method = "glm", Accuracy =glm_acc , Sensitivity=glm_sens,
                                                     Precision = glm_prec)
                                           )

results%>%  knitr::kable(col.names = c("Model","Accuracy","Sensitivity","Precision"),"simple")

####################################################################################
# Model 3 - lda
####################################################################################

# set seed for consistent results
set.seed(1, sample.kind = "Rounding") 

# Traing using lda model
train_lda <- train(train_x, train_y$target, method = "lda",
                   trControl = ctrl)

# get predictions on test set
lda_preds <- predict(train_lda, test_x)

# lda accuracy,  sensitivity, precision
lda_acc <- confusionMatrix(lda_preds, test_y$target, positive ="yes")$overall["Accuracy"]  
lda_sens <- sensitivity(lda_preds, test_y$target, positive ="yes")
lda_prec <- precision(lda_preds, test_y$target, relevant ="yes")

# Add lda results to the table
results <- bind_rows(results,
                                          data_frame(method = "lda", Accuracy =lda_acc , Sensitivity=lda_sens,
                                                     Precision = lda_prec)
                                          )

results%>%  knitr::kable(col.names = c("Model","Accuracy","Sensitivity","Precision"),"simple")

####################################################################################
# Model 4-5 - Apply down sampling technique for glm, lda 
# https://www.r-bloggers.com/2016/12/handling-class-imbalance-with-r-and-caret-an-introduction/
####################################################################################
 
# down sample the prevalent no class and then train glm, lda  models again
ctrl$sampling <- "down"

 # set seed for consistent results
 set.seed(1, sample.kind = "Rounding") 
 
# Model 4 - glm with down sampling
train_glm_d <- train(train_x, train_y$target, method = "glm",
                   trControl = ctrl)

# get predictions on test set
glm_preds_d <- predict(train_glm_d, test_x)

# glm accuracy,  sensitivity, precision
glm_acc_d <- confusionMatrix(glm_preds_d, test_y$target, positive ="yes")$overall["Accuracy"]  
glm_sens_d <- sensitivity(glm_preds_d, test_y$target, positive ="yes")
glm_prec_d <- precision(glm_preds_d, test_y$target, relevant ="yes")

# Add glm+down sample results to the table
 results <- bind_rows(results,
                                          data_frame(method = "glm + down sample", 
                                                     Accuracy =glm_acc_d , Sensitivity=glm_sens_d,
                                                     Precision = glm_prec_d)
                                          )
 results%>%  knitr::kable(col.names = c("Model","Accuracy","Sensitivity","Precision"),"simple")
 
 # set seed for consistent results
 set.seed(1, sample.kind = "Rounding") 

# Model 5- Train using lda model with down sampling
train_lda_d <- train(train_x, train_y$target, method = "lda",
                   trControl = ctrl)

# get predictions on test set
lda_preds_d <- predict(train_lda_d, test_x)

#  lda accuracy,  sensitivity, precision
lda_acc_d <- confusionMatrix(lda_preds_d, test_y$target, positive ="yes")$overall["Accuracy"]  
lda_sens_d <- sensitivity(lda_preds_d, test_y$target, positive ="yes")
lda_prec_d <- precision(lda_preds_d, test_y$target, relevant ="yes")

# Add lda + downsample results to the table
results <- bind_rows(results,
                                          data_frame(method = "lda + down sample", Accuracy =lda_acc_d ,
                                                     Sensitivity=lda_sens_d,
                                                     Precision = lda_prec_d)
)

# output the results with downsampling
results%>%  knitr::kable(col.names = c("Model","Accuracy","Sensitivity","Precision"),"simple")


######################################################################
# Test final selected model against validation set or the final hold out set
######################################################################

# standardize the variables 
v_x_centered <- sweep(validation_x, 2, colMeans(validation_x))
v_x_scaled <- sweep(v_x_centered, 2, colSds(validation_x), FUN = "/")

# get predictions on test set
validation_preds <- predict(train_glm_d, v_x_scaled)

validation_acc<- confusionMatrix(validation_preds, validation_y$target, positive ="yes")$overall["Accuracy"]  
validation_sens <- sensitivity(validation_preds, validation_y$target, positive ="yes")
validation_prec <- precision(validation_preds, validation_y$target, relevant ="yes")

validation_results <- data_frame(method = "Validation Results - glm + down sample", Accuracy =validation_acc , Sensitivity=validation_sens,
                                 Precision = validation_prec)
validation_results%>%  knitr::kable(col.names = c("Model","Accuracy","Sensitivity","Precision"),"simple")


