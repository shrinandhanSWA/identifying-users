# This is the R code that contains the original classifier


#Machine Learning
rm(list=ls())

#Libraries needed
library('dplyr')
library('randomForest')
library('pander')
library('caret')

#Information
pander(sessionInfo())

#**R version 4.0.1 (2020-06-06)**
  
#**Platform:** x86_64-apple-darwin17.0 (64-bit) 

#**locale:**
  #en_GB.UTF-8||en_GB.UTF-8||en_GB.UTF-8||C||en_GB.UTF-8||en_GB.UTF-8

#**attached base packages:** 
  #_stats_, _graphics_, _grDevices_, _utils_, _datasets_, _methods_ and _base_

#**other attached packages:** 
 #_pander(v.0.6.3)_, _randomForest(v.4.6-14)_ and _dplyr(v.1.0.2)_

#**loaded via a namespace (and not attached):** 
  #_Rcpp(v.1.0.5)_, _digest(v.0.6.26)_, _crayon(v.1.3.4)_, _R6(v.2.4.1)_, _lifecycle(v.0.2.0)_, _magrittr(v.1.5)_, _pillar(v.1.4.6)_, _rlang(v.0.4.8)_, _rstudioapi(v.0.11)_, _vctrs(v.0.3.4)_, _generics(v.0.0.2)_, _ellipsis(v.0.3.1)_, _tools(v.4.0.1)_, _glue(v.1.4.2)_, _purrr(v.0.3.4)_, _yaml(v.2.2.1)_, _compiler(v.4.0.1)_, _pkgconfig(v.2.0.3)_, _tidyselect(v.1.1.0)_ and _tibble(v.3.0.4)_

#Read In Data


# base_dir is where this R file is (working directory is not automatically set, somehow)
base_dir = gsub(" ", "", paste(getwd(), '/identifying-users'))   # change this as required!
setwd(base_dir)


BehavProfilesPickupsData <- read.csv("csv_files\\PickupsBehavProf.csv", header = T)
BehavProfilesDurationData <- read.csv("csv_files\\TimeBehavProf.csv", header = T)
BehavProfilesDurationData46 <- read.csv("csv_files\\TimeBehavProf46.csv", header = T)
BehavProfilesDurationDataWeekday <- read.csv("csv_files\\TimeBehavProfWeekday.csv", header = T)
BehavProfilesDurationDataWeekend <- read.csv("csv_files\\TimeBehavProfWeekend.csv", header = T)
BehavProfilesDurationDataWeekdayApps <- read.csv("csv_files\\TimeBehavProfWeekdayAllApps.csv", header = T)
BehavProfilesDurationData12hBins <- read.csv("csv_files\\TimeBehavProf12hBins.csv", header = T)


#Make person number a factor (so can be classified) 
BehavProfilesPickupsData$Person <- as.factor(as.character(BehavProfilesPickupsData$Person))
BehavProfilesDurationData$Person <- as.factor(as.character(BehavProfilesDurationData$Person))
BehavProfilesDurationData46$Person <- as.factor(as.character(BehavProfilesDurationData46$Person))
BehavProfilesDurationDataWeekday$Person <- as.factor(as.character(BehavProfilesDurationDataWeekday$Person))
BehavProfilesDurationDataWeekend$Person <- as.factor(as.character(BehavProfilesDurationDataWeekend$Person))
BehavProfilesDurationDataWeekdayApps$Person <- as.factor(as.character(BehavProfilesDurationDataWeekdayApps$Person))
BehavProfilesDurationData12hBins$Person <- as.factor(as.character(BehavProfilesDurationData12hBins$Person))



#Separating the Data 

#Pickups
# PickupsTrain <- select(filter(BehavProfilesPickupsData, Day %in% c("Day1", "Day2", "Day3", "Day4", "Day5", "Day6")), -Day) 
# PickupsTest <- select(filter(BehavProfilesPickupsData, Day == "Day7"), -Day) 

#Durations
# DurationsTrain <- select(filter(BehavProfilesDurationData, Day %in% c("Day1", "Day2", "Day3", "Day4", "Day5", "Day6")), -Day) 
# DurationsTest <- select(filter(BehavProfilesDurationData, Day == "Day7"), -Day) 

#Small Durations
#Basically, the -Day drops the Day column. The filtering only gives days in that range, so Day8 and above is ignored
# DurationsTrain46 <- select(filter(BehavProfilesDurationData46, Day %in% c("Day1", "Day2", "Day3", "Day4", "Day5", "Day6")), -Day) 
# DurationsTest46 <- select(filter(BehavProfilesDurationData46, Day == "Day7"), -Day) 

split_day = 'Day1'

# Better splitting
DurationsTrain46 <- select(filter(BehavProfilesDurationData46, Day != split_day), -Day) 
DurationsTest46 <- select(filter(BehavProfilesDurationData46, Day == split_day), -Day) 


print(' ')

#Weekdays Durations - idk if this will work
# DurationsTrainWeekday <- select(filter(BehavProfilesDurationDataWeekday, Day %in% c("Day1", "Day2", "Day3", "Day4", "Day5", "Day6")), -Day) 
# DurationsTestWeekday <- select(filter(BehavProfilesDurationDataWeekday, Day == "Day7"), -Day) 

# DurationsTrainWeekend <- select(filter(BehavProfilesDurationDataWeekend, Day %in% c("Day1", "Day2", "Day3", "Day4", "Day5", "Day6")), -Day) 
# DurationsTestWeekend <- select(filter(BehavProfilesDurationDataWeekend, Day == "Day7"), -Day) 

# Better splitting - modified data
DurationsTrainWeekday <- select(filter(BehavProfilesDurationDataWeekday, Day != split_day), -Day) 
DurationsTestWeekday <- select(filter(BehavProfilesDurationDataWeekday, Day == split_day), -Day) 

DurationsTrainWeekend <- select(filter(BehavProfilesDurationDataWeekend, Day != split_day), -Day) 
DurationsTestWeekend <- select(filter(BehavProfilesDurationDataWeekend, Day == split_day), -Day) 

DurationsTrainWeekdayApps <- select(filter(BehavProfilesDurationDataWeekdayApps, Day != split_day), -Day) 
DurationsTestWeekdayApps <- select(filter(BehavProfilesDurationDataWeekdayApps, Day == split_day), -Day) 

DurationsTrain12h <- select(filter(BehavProfilesDurationData12hBins, Day != split_day), -Day) 
DurationsTest12h <- select(filter(BehavProfilesDurationData12hBins, Day == split_day), -Day) 


print("nandhu prabu")

#Random Forest Modelling - Will take a good five minutes to train each forest
set.seed(2020)

#Pickups Forest
# RForestPickups <- randomForest::randomForest(Person ~ ., data = PickupsTrain, ntree = 3120, importance=TRUE)

#Durations Forest
# RForestDurations <- randomForest::randomForest(Person ~ ., data = DurationsTrain, ntree = 3120, importance=TRUE)

#Small Durations Forest
RForestDurations46 <- randomForest::randomForest(Person ~ ., data = DurationsTrain46, ntree = 3120, importance=TRUE) # the importance argument just changes the plots, idk

#Weekdays Durations Forest
RForestDurationsWeekday <- randomForest::randomForest(Person ~ ., data = DurationsTrainWeekday, ntree = 3120, importance=TRUE)

#Weekends Durations Forest
RForestDurationsWeekend <- randomForest::randomForest(Person ~ ., data = DurationsTrainWeekend, ntree = 3120, importance=TRUE)

#Weekdays + All Apps Forest
RForestDurationsWeekdayApps <- randomForest::randomForest(Person ~ ., data = DurationsTrainWeekdayApps, ntree = 3120, importance=TRUE)

#Weekdays + popular apps + 12h time bins
RForestDurations12h <- randomForest::randomForest(Person ~ ., data = DurationsTrain12h, ntree = 3120, importance=TRUE)



#Examining the Forests

#Pickups Forest summary
# print(RForestPickups)
# varImpPlot(RForestPickups)

#Durations Forest summary
# print(RForestDurations)
# varImpPlot(RForestDurations)

#Small Durations Forest summary
# print(RForestDurations46)
# varImpPlot(RForestDurations46)

#Accuracy of forest predictions on training data - Pickups
# RForestPickupsPredictions <- predict(RForestPickups, PickupsTrain, type = 'class')
# PickupsConfusionMatrix <- table(PickupsTrain$Person, RForestPickupsPredictions)
# PickupsAccuracy <- sum(diag(PickupsConfusionMatrix))/sum(PickupsConfusionMatrix)
# cat("Accuracy (%):", PickupsAccuracy)

#Accuracy of forest predictions on training data - Durations
# RForestDurationsPredictions <- predict(RForestDurations, DurationsTrain, type = 'class')
# DurationsConfusionMatrix <- table(DurationsTrain$Person, RForestDurationsPredictions)
# DurationsAccuracy <- sum(diag(DurationsConfusionMatrix))/sum(DurationsConfusionMatrix)
# cat("Accuracy (%):", DurationsAccuracy)

#Accuracy of forest prediction on small training data - Durations
# RForestDurationsPredictions46 <- predict(RForestDurations46, DurationsTrain46, type = 'class')
# DurationsConfusionMatrix46 <- table(DurationsTrain46$Person, RForestDurationsPredictions46)
# DurationsAccuracy46 <- sum(diag(DurationsConfusionMatrix46))/sum(DurationsConfusionMatrix46)
# cat("Accuracy (%):", DurationsAccuracy46)

#Accuracy of forest predictions on test data - Pickups
# RForestPickupsPredictionsTest <- predict(RForestPickups, PickupsTest, type = 'class')
# PickupsConfusionMatrixTest <- table(PickupsTest$Person, RForestPickupsPredictionsTest)
# PickupsAccuracyTest <- sum(diag(PickupsConfusionMatrixTest))/sum(PickupsConfusionMatrixTest)
# cat("Accuracy (%):", PickupsAccuracyTest)

#Accuracy of forest predictions on test data - Durations
# RForestDurationsPredictionsTest <- predict(RForestDurations, DurationsTest, type = 'class')
# DurationsConfusionMatrixTest <- table(DurationsTest$Person, RForestDurationsPredictionsTest)
# DurationsAccuracyTest <- sum(diag(DurationsConfusionMatrixTest))/sum(DurationsConfusionMatrixTest)
# cat("Accuracy (%):", DurationsAccuracyTest)

#Accuracy of forest predictions on small test data - Durations
RForestDurationsPredictionsTest46 <- predict(RForestDurations46, DurationsTest46, type = 'class')
DurationsConfusionMatrixTest46 <- table(DurationsTest46$Person, RForestDurationsPredictionsTest46)
DurationsAccuracyTest46 <- sum(diag(DurationsConfusionMatrixTest46))/sum(DurationsConfusionMatrixTest46)
str1 = paste("Original method accuracy (%): ", DurationsAccuracyTest46*100)

# Accuracy of forest predictions on weekday test data - Durations
RForestDurationsPredictionsTestWeekday <- predict(RForestDurationsWeekday, DurationsTestWeekday, type = 'class')
DurationsConfusionMatrixTestWeekday <- table(DurationsTestWeekday$Person, RForestDurationsPredictionsTestWeekday)
DurationsAccuracyTestWeekday <- sum(diag(DurationsConfusionMatrixTestWeekday))/sum(DurationsConfusionMatrixTestWeekday)
str2 = paste("  Weekdays only accuracy (%): ", DurationsAccuracyTestWeekday*100)

# Accuracy of forest predictions on weekend test data - Durations
RForestDurationsPredictionsTestWeekend <- predict(RForestDurationsWeekend, DurationsTestWeekend, type = 'class')
DurationsConfusionMatrixTestWeekend <- table(DurationsTestWeekend$Person, RForestDurationsPredictionsTestWeekend)
DurationsAccuracyTestWeekend <- sum(diag(DurationsConfusionMatrixTestWeekend))/sum(DurationsConfusionMatrixTestWeekend)
str3 = paste("  Weekends only accuracy (%): ", DurationsAccuracyTestWeekend*100)

# Accuracy of forest predictions on weekday + all apps test data - Durations
RForestDurationsPredictionsTestWeekdayApps <- predict(RForestDurationsWeekdayApps, DurationsTestWeekdayApps, type = 'class')
DurationsConfusionMatrixTestWeekdayApps <- table(DurationsTestWeekdayApps$Person, RForestDurationsPredictionsTestWeekdayApps)
DurationsAccuracyTestWeekdayApps <- sum(diag(DurationsConfusionMatrixTestWeekdayApps))/sum(DurationsConfusionMatrixTestWeekdayApps)
str4 = paste("  Using all apps (weekdays) (%):", DurationsAccuracyTestWeekdayApps*100)


# Accuracy of forest predictions on 12h bins (all days, popular apps) - Durations
RForestDurationsPredictionsTest12h <- predict(RForestDurations12h, DurationsTest12h, type = 'class')
DurationsConfusionMatrixTest12h <- table(DurationsTest12h$Person, RForestDurationsPredictionsTest12h)
DurationsAccuracyTest12h <- sum(diag(DurationsConfusionMatrixTest12h))/sum(DurationsConfusionMatrixTest12h)
str5 = paste("  12h bins (%):", DurationsAccuracyTest12h*100)

RForestDurations12h


print(paste(split_day, ":  ", str1, str2, str3, str4, str5))

# #Calculate probabilities for test data - Pickups
# probabilitiesPickupsTest <- as.data.frame(predict(RForestPickups, PickupsTest, type = "prob"))
# probabilitiesPickupsTest <- probabilitiesPickupsTest[c(as.character(1:780))]
# countPickupsTest <- probabilitiesPickupsTest*RForestPickups$ntree

# #Calculate probabilities for test data - Durations
# probabilitiesDurationsTest <- as.data.frame(predict(RForestDurations, DurationsTest, type = "prob"))
# probabilitiesDurationsTest <- probabilitiesDurationsTest[c(as.character(1:780))]
# countDurationsTest <- probabilitiesDurationsTest*RForestDurations$ntree

# #Rank the probabilities - Pickups
# rankorderprobPickups <- apply(probabilitiesPickupsTest, 2, rank)

# #Rank the probabilities - Durations
# rankorderprobDurations <- apply(probabilitiesDurationsTest, 2, rank)


# #Write function that searches for the top 10 ranks, and converts them to TRUE

# topXtrue <- function (dataforfunction) { 
#   Criteria <- length(dataforfunction) - 10
#   test <- dataforfunction >= Criteria
# }


# #Run function on Pickups data
# GetTruesPickups <- apply(rankorderprobPickups, 2, topXtrue)

# #Run function on Duration data
# GetTruesDurations <- apply(rankorderprobDurations, 2, topXtrue)

# #Calculate the % of times the correct person is in the top 10 people (Pickups)
# sum(diag(GetTruesPickups) == TRUE) / nrow(GetTruesPickups)

# #Calculate the % of times the correct person is in the top 10 people (Durations)
# sum(diag(GetTruesDurations) == TRUE) / nrow(GetTruesDurations)

# #Explore other performance measures
# StatisticsPickupsTest <- confusionMatrix(RForestPickupsPredictionsTest, PickupsTest$Person)
# StatisticsDurationsTest <- confusionMatrix(RForestDurationsPredictionsTest, DurationsTest$Person)
# ByClassPickups <- as.data.frame(StatisticsPickupsTest$byClass)
# ByClassDurations <- as.data.frame(StatisticsPickupsTest$byClass)

# #calc means
# apply(ByClassPickups, 2, mean)
# apply(ByClassDurations, 2, mean)

# #Files to save
# write.csv(DurationsConfusionMatrixTest, "DurationsConfusionMatrixTest.csv")
# write.csv(PickupsConfusionMatrixTest, "PickupsConfusionMatrixTest.csv")
# write.csv(GetTruesPickups, "Top10PickupsConfusionMatrixTest.csv")
# write.csv(GetTruesDurations, "Top10DurationsConfusionMatrixTest.csv")
# write.csv(ByClassPickups, "PickupsPerformanceMeasuresPerPerson.csv")
# write.csv(ByClassDurations, "DurationsPerformanceMeasurePerPerson.csv")
