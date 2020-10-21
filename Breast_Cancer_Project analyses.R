#Breast Cancer Project - edX HarvardX


#The brca dataset from dslabs package is used,
#which contains information about breast cancer diagnosis biopsy samples
#for tumors that were determined to be either benign and malignant.
  
#brca$y: a vector of sample classifications ("B" = benign or "M" = malignant)
#brca$x: a matrix of numeric features describing properties of the shape and size of cell nuclei extracted from biopsy microscope images

#loading the data by setting options and loading the libraries
options(digits = 3)
library(matrixStats)
library(tidyverse)
library(caret)
library(dslabs) #make sure this package is updated by typing: install.packages("dslabs")
data(brca)

#Ass18 Q1 
str(brca) #check the structure of the database
dim(brca$x)[1] #determine the number of samples
dim(brca$x)[2] #determine the number of features
mean(brca$y == "M") #determine the proportion of malignant samples
which.max(colMeans(brca$x)) #determine the column number with the highest mean
which.min(colSds(brca$x)) #determine the column number with the lowest sd

#Ass18 Q2
#define objects x and y for more concise coding
x <- brca$x
y <- brca$y

scale1 <- sweep(x, 2, colMeans(x), FUN = "-")#scale each column by subtracting the column mean
class(scale1)
scale2 <- sweep(scale1, 2, colSds(scale1), FUN = "/") #rescale by dividing by the column SD.
sd(scale2[,1]) #determine SD of the 1st column
median(scale2[,1]) #determine the median of the 1st column

#Ass18 Q3
#Calculate the distance between all SAMPLES using the scaled matrix (= "scale2"):
dist3 <- dist(scale2)
dist3 <- as.matrix(dist3) #convert dist object into matrix

as.matrix(dist3[ind_B,]) [1:5, 1:5] #explore 1st 5 rows and columns to have a better picture of the matrix symmetry
#Note: 1st row equals the 1st column etc.

#Define indices of benign and malignant samples, respectively:
ind_B <- which(y == "B")
ind_M <- which(y == "M")

#Determine average distances between 1st sample and other benign samples:
vektor_dist <- dist3[1,] #Only 1st row is extracted, since we're only interested in comparisons to the 1st sample
mean(vektor_dist[ind_B])
#Determine average distances between 1st sample and malignant samples:
mean(vektor_dist[ind_M])

#Ass18 Q4
#Make a heatmap of the relationship between FEATURES using the scaled matrix.
#Since we are interested in relations between features, not samples,
#we need to transpose the matrix of dist!!! (see the textbook, p627):
d_features <- dist(t(scale2))
Heatmap_features <- heatmap(as.matrix(d_features), labRow = NA, labCol = NA)
Heatmap_features

#Save this plot as the png file:
png(filename="Figs/Heatmap_features.png")
heatmap(as.matrix(d_features), labRow = NA, labCol = NA)
dev.off()

#Ass18 Q5
#Perform hierarchical clustering and cluster dendrogram:
h <- hclust(d_features)
plot(h, cex = 0.65)
#Save the cluster dendrogram as png file:
png(filename="Figs/Cluster_dendrogram_features.png")
plot(h, cex = 0.65)
dev.off()

#Cut the tree into 5 groups (see the textbook, p679):
groups <- cutree(h, k = 5)
split(names(groups), groups)

#Ass18 Q6
#Perform a principal component analysis of the scaled matrix:
pca <- prcomp(scale2)
#Determine the proportion of variance of PCA1; use summary (see the textbook, p639):
summary(pca)

#Ass18 Q7
#Plot the first two principal components with color representing tumor type B/M
class(pca$x[,1:2]) #confirm that the class of the pca object is a matrix, which needs to be changed into dataframe before plotting:
data.frame(pca$x[,1:2], y) %>%
  ggplot(aes(PC1, PC2, col = y))+
  geom_point()
#Save the ggplot as png file:
ggsave("Figs/PC2vsPC1_for_tumor_type.png")

#Ass18 Q8
#Make a boxplot of the first 10 PCs grouped by tumor type.
#Solution with a for loop, used in some other similar examples:
for(i in 1:10){
  boxplot(pca$x[,i] ~ y, main = paste("PC", i))
} #the downside it that each PC is plotted on a separate boxplot.

#Better solution, which plots all the boxplots on the same figure:
data.frame(type = brca$y, pca$x[,1:10]) %>%
  gather(key = "PC", value = "value", -type) %>%
  ggplot(aes(PC, value, fill = type)) +
  geom_boxplot() 
#Save the ggplot as png file:
ggsave("Figs/Boxplot_10PCs_by_tumor_type.png")

#Ass18 Q9
x_scaled <- scale2 #Rename scale2 to follow the following example in the edX course

#Set the seed to 1, then create a data partition splitting brca$y
#and the scaled version of the brca$x matrix into a 20% test set and 80% train:
set.seed(1) # if using R 3.5 or earlier
#set.seed(1, sample.kind = "Rounding")    # if using R 3.6 or later
test_index <- createDataPartition(brca$y, times = 1, p = 0.2, list = FALSE)
test_x <- x_scaled[test_index,]
test_y <- brca$y[test_index]
train_x <- x_scaled[-test_index,]
train_y <- brca$y[-test_index]

#determine the training set proportion of benign tumors:
train <- data.frame(train_x, type = train_y)
mean(train$type == "B")
#determine the test set proportion of benign tumors:
test <- data.frame(test_x, type = test_y)
mean(test$type == "B")

#Save test and train datasets as files:
  save(train, file = "rData/TrainDataset.rda")
  save(test, file = "rData/TestDataset.rda")
  
  
#Ass18 Q10a
#Function predict_kmeans() is predefined in the course.
#It takes two arguments - a matrix of observations x and a k-means object k
  #and assigns each row of x to a cluster from k.
predict_kmeans <- function(x, k) {
    centers <- k$centers    # extract cluster centers
    # calculate distance to cluster centers
    distances <- sapply(1:nrow(x), function(i){
      apply(centers, 1, function(y) dist(rbind(x[i,], y)))
    })
    max.col(-t(distances))  # select cluster with min distance to center
  }

#Setting the seed to 3:
set.seed(3)
#Perform k-means clustering (textbook, p681) on the training set with 2 centers and assign the output to k:
k <- kmeans(train_x, centers = 2)
#Use the predict_kmeans() to make predictions on the test set:
kmeans_preds <- predict_kmeans(test_x, k)
#Change numerical output (1,2) into characters (B, M):
kmeans_preds_char <- ifelse(kmeans_preds == 1, "B", "M")
#Calculate the overall accuracy:
mean(kmeans_preds_char == test_y)

#Ass18 Q10b
#Determine the proportion of benign tumors correctly identified:
test_y_char <- as.character(test_y)
df_char <- as.data.frame(cbind(KMf = kmeans_preds_char, Yf = test_y_char))
all_YBs <- df_char %>% mutate(matches = KMf == Yf) %>% filter(Yf == "B")
mean(all_YBs$matches)
#The same with the use of sensitivity() function:
sensitivity(factor(kmeans_preds_char), test_y, positive = "B")

#Determine the proportion of malignant tumors correctly identified:
all_YMs <- df_char %>% mutate(matches = KMf == Yf) %>% filter(Yf == "M")
mean(all_YMs$matches)
#The same with the use of sensitivity() function:
sensitivity(factor(kmeans_preds_char), test_y, positive = "M")

#Ass18 Q11
#Fit a logistic regression model on the training set using all predictors.
fit_glm <- train(train_x, train_y, method = "glm")

#Obtain predictors and accuracy:
y_hat_glm <- predict(fit_glm, test_x)
confusionMatrix(data = y_hat_glm, reference = test_y)$overall["Accuracy"]

#Ass18 Q12
#Train an LDA model on the training set and make predictions:
fit_lda <- train(train_x, train_y, method = "lda")
y_hat_lda <- predict(fit_lda, test_x)
confusionMatrix(data = y_hat_lda, reference = test_y)$overall["Accuracy"]

#Train an QDA model on the training set and make predictions:
fit_qda <- train(train_x, train_y, method = "qda")
y_hat_qda <- predict(fit_qda, test_x)
confusionMatrix(data = y_hat_qda, reference = test_y)$overall["Accuracy"]

#Ass18 Q13
library(gam) #load the gam package

set.seed(5) #Set the seed to 5
#Fit a loess model on the training set, with default tuning:
fit_loess <- train(train_x, train_y, method = "gamLoess") #NOTE: ignore warnings.
#Generate predictions:
y_hat_loess <- predict(fit_loess, test_x)
confusionMatrix(data = y_hat_loess, reference = test_y)$overall["Accuracy"]

#Ass18 Q14
#Train a k-nearest neighbors model on the training set, with odd values of  k  from 3 to 21.
k <- seq(3,21,2)
set.seed(7) #Set the seed to 7
fit_knn <- train(train_x, train_y, method = "knn", tuneGrid = data.frame(k = k))
#Determine which is the final k value used in the model:
fit_knn$bestTune
#A plot that shows the accuracy of the model based on different k values:
plot(fit_knn)

#Generate predictions:
y_hat_knn <- predict(fit_knn, test_x)
confusionMatrix(data = y_hat_knn, reference = test_y)$overall["Accuracy"]

#Ass18 Q15
#Train a random forest model on the training set, with mtry values of 3, 5, 7 and 9.
#Also, use the argument importance = TRUE so that feature importance can be extracted.
set.seed(9) #Set the seed to 9
fit_rf <- train(train_x, train_y, method = "rf",tuneGrid = data.frame(mtry = c(3,5,7,9)), importance = TRUE)

#Generate predictions:
y_hat_rf <- predict(fit_rf, test_x)
fit_rf$bestTune #Determine the best mtry value used in the model
plot(fit_rf) #A plot that shows the accuracy of the model based on different mtry values:
confusionMatrix(data = y_hat_rf, reference = test_y)$overall["Accuracy"] #overall accuracy

#Determine the most important variable in the random forest model:
varImp(fit_rf)

#Ass18 Q16a
#We want to create an ensemble (the average of predictions) using the predictions from all models,
#except k-means.
#First, we create a matrix with all predictions:
matrix_ensemble <- cbind(y_hat_glm, y_hat_lda, y_hat_qda, y_hat_loess, y_hat_knn, y_hat_rf)

row_avg <- rowMeans(matrix_ensemble) #calculate row means
#Create the ensemble prediction by a logical expression from the row means
#(if most models suggest the tumor is malignant, predict malignant):
y_hat_ensemble <- ifelse(row_avg <= 1.5, "B", "M")

#Determining the accuracy of the ensemble predictions:
confusionMatrix(data = as.factor(y_hat_ensemble), reference = test_y)$overall["Accuracy"]

#Ass18 Q16b
#Let's make a table of all the accuracies, incl. k_means:
as.factor(kmeans_preds_char)
models <- c("K-means", "Logistic regression", "LDA", "QDA", "Loess", "KNN", "Random forest", "Ensemble")
all_acc <- c(mean(kmeans_preds_fact == test_y),
             mean(y_hat_glm == test_y),
             mean(y_hat_lda == test_y),
             mean(y_hat_qda == test_y),
             mean(y_hat_loess == test_y),
             mean(y_hat_knn == test_y),
             mean(y_hat_rf == test_y),
             mean(as.factor(y_hat_ensemble) == test_y))
data.frame(Model = models, Accuracy = all_acc) %>% arrange(desc(Accuracy))





