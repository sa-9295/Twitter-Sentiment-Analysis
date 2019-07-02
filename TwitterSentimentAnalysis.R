#Importing libraries
library(dplyr)
library(caret)
library(ggplot2)
library(tm) #for text mining
library(SnowballC) #for stemming
#install.packages("wordcloud") # word-cloud generator 
library(wordcloud)
#install.packages("RColorBrewer") # color palettes
library(RColorBrewer)

#Importing datasets
training_set = read.csv('train_E6oV3lV.csv',stringsAsFactors = FALSE)
test_set  = read.csv('test_tweets_anuFYb8.csv',stringsAsFactors = FALSE)
submission = read.csv("sample_submission_gfvA5FD.csv",stringsAsFactors = FALSE)

#Features of Data
names(training_set)
names(test_set)

#Structure of Data
str(training_set)
str(test_set)

#Let's look for label distribution in training dataset
table(training_set$label)
#We have 2242 tweets labeled as racist or sexist and 29720 tweets labeled as non-racist/sexist.

#Combining training and test
test_set$label =NA
dataset = rbind(training_set,test_set)

#Data Cleaning
corpus = VCorpus(VectorSource(dataset$tweet))
corpus <- tm_map(corpus, content_transformer(function(x) iconv(enc2utf8(x), sub = "byte")))
#Convert to lower case
corpus <- tm_map(corpus, content_transformer(tolower))

#Removing any character not within the range of x20 (SPACE) and x7E (~)
TextProcessing1 <-function(x){
  gsub('[^\x20-\x7E]', '', x)
}
corpus= tm_map(corpus, content_transformer(TextProcessing1))

#We will remove twitter handles as they don't give any information about the nature of tweet
TextProcessing2 <-function(x){
  gsub("https.*","",x) ## Remove URLs
  gsub("http.*","",x) ## Remove URLs
  gsub('[[:cntrl:]]', '', x) ## Remove Controls characters
  gsub("@\\S+",'',x) ## Remove @user
}
corpus= tm_map(corpus, content_transformer(TextProcessing2))

#Get rid of the punctuations, numbers, special characters , extra spaces
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus,stripWhitespace)

#Remove stop words
corpus = tm_map(corpus, removeWords, stopwords("english"))

#Stemming i.e. normalize data to their root word
corpus = tm_map(corpus, stemDocument)

#Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm,0.999)
dataset_final = as.data.frame(as.matrix(dtm))
dataset_final$label = dataset$label
# tdm = TermDocumentMatrix(corpus)
dataset_1 = as.matrix(dtm)
dataset_2 = sort(colSums(dataset_1),decreasing=TRUE)
dataset_refined = data.frame(word=names(dataset_2),freq=dataset_2)
head(dataset_refined,10)

#Generating the word cloud
set.seed(1234)
wordcloud(words=dataset_refined$word,freq=dataset_refined$freq,min.freq = 1,max.words = 200,
          random.order = FALSE, scale=c(3,0.5),rot.per = 0.35, colors=brewer.pal(8,"Dark2"))
#We can see that most of the words are neutral. Words like love, happi, day, thank are the most frequent ones
# It doesn't give us any idea about the words associated with the racist/sexist tweets. 
#Hence, we will plot separate wordclouds for both the classes (racist/sexist or not) in our train data.

#Words in Non-racist/sexist tweets
normal_words = dataset_final[which(dataset_final$label == 0),]
nw1 = sort(colSums(normal_words),decreasing=TRUE)
normal_words_data = data.frame(word=names(nw1),freq=nw1)
head(normal_words_data)
#Generating the word cloud
set.seed(1234)
wordcloud(words=normal_words_data$word,freq=normal_words_data$freq,min.freq = 1,max.words = 200,
          random.order = FALSE, scale=c(3,0.5),rot.per = 0.35, colors=brewer.pal(8,"Dark2"))

#Words in Racist/sexist tweets
negative_words = dataset_final[which(dataset_final$label == 1),]
nw2 = sort(colSums(negative_words),decreasing=TRUE)
negative_words_data = data.frame(word=names(nw2),freq=nw2)
head(negative_words_data)
#Generating the word cloud
set.seed(1234)
wordcloud(words=negative_words_data$word,freq=negative_words_data$freq,min.freq = 1,max.words = 200,
          random.order = FALSE, scale=c(3,0.5),rot.per = 0.35, colors=brewer.pal(8,"Dark2"))

# Encoding the target feature as factor
dataset_final$label = factor(dataset_final$label, levels = c(0, 1))

#Separating the train adn test sets
training_set = dataset_final[1:31962,]
test_set = dataset_final[31963:nrow(dataset_final),]

#Modeling
# Splitting the dataset into the Training set and Validation set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(training_set$label, SplitRatio = 0.75)
train = subset(training_set, split == TRUE)
validation = subset(training_set, split == FALSE)

#Logistic Regression

# With glm it is giving me a warning glm.fit: fitted probabilities numerically 0 or 1 occurred
#It means your problem has separation or quasi-separation (a subset of the data that is predicted perfectly and may be running a subset of the coefficients out to infinity).
#Let's try Firth's Bias-Reduced Logistic Regression
#Confidence intervals for regression coefficients can be computed by penalized profile likelihood. Firth's method was proposed as ideal solution to the problem of separation in logistic regression.
#install.packages("logistf")
#library(logistf)
# classifier = logistf(formula = label~.,  data = train)
# prod_pred = predict(classifier, type="response",newdata=validation[-(which(colnames(validation)=="label"))])
# y_pred=ifelse(prod_pred>0.5, 1,0)

#With logistf it is taking forever to run
#Let's try with glmnet
library(glmnet)
#cross validation glmnet
# The function runs glmnet nfolds+1 times; the first to get the lambda sequence, and then the remainder to compute the fit with each of the folds omitted. 
classifier = cv.glmnet(x=as.matrix(train[,-(which(colnames(train)=="label"))]),y=train$label,family="binomial",
                    alpha =1, nfolds=5)
#Make predictions
prod_pred = predict(classifier, type="response",newx=as.matrix(validation[,-1108]),
                    s="lambda.min")
y_pred=ifelse(prod_pred>0.5, 1,0)

cm=table(validation[,(which(colnames(validation)=="label"))],y_pred)
#install.packages("MLmetrics")
library(MLmetrics)
F1_Score(y_true = validation$label,y_pred=y_pred)

#Predicting the test set
prod_pred = predict(classifier, type="response",newx=as.matrix(test_set[,-(which(colnames(test_set)=="label"))]))
test_pred=ifelse(prod_pred>0.5, 1,0)
submission$label = test_pred
write.csv(submission, file = "Logistic Result.csv",row.names = FALSE)


#Support Vector Machine with grid search
library(e1071)
set.seed(1234)
my_control=trainControl(method="cv",number =5)
tgrid = expand.grid(C = c(2,1,1.75,1.5,0.5,0.1),
                    sigma = c(0.01,0.025,0.05,0.1,0.25,0.5,0.75))
classifier = train(x=training_set[,-(which(colnames(training_set)=="label"))],
                     y=training_set$label, method="svmRadial",
                     trControl = my_control, tuneGrid = tgrid)
plot(classifier)
classifier$bestestimator
y_pred = predict(classifier, newdata = test_set[,-1108])
test_pred=ifelse(prod_pred>0.5, 1,0)
submission$label = test_pred
write.csv(submission, file = "SVM Result.csv",row.names = FALSE)

#Random Forest
set.seed(1234)
my_control=trainControl(method="cv",number =5)
tgrid = expand.grid(.mtry=seq(5,32,by=3), .splitrule = "gini", .min.node.size = c(10,15,20))
classifier = train(x=training_set[,-(which(colnames(training_set)=="label"))],
                     y=training_set$label, method="rf",
                     trControl = my_control, tuneGrid = tgrid,
                     num.trees=500, n_jobs=-1, oob_score=TRUE)
plot(classifier) # We can see best score at mtry =  and min.node.size = 
y_pred = predict(classifier, newdata = test_set[,-1108])
test_pred=ifelse(prod_pred>0.5, 1,0)
submission$label = test_pred
write.csv(submission, file = "Random Forest Result.csv",row.names = FALSE)