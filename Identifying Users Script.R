# This is the R code that contains the original classifier


#Machine Learning
rm(list=ls())

#Libraries needed
library('dplyr')
library('randomForest')
library('pander')
library('caret')
library('sjmisc')
library('stringr')

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


# On my laptop, when a new R terminal starts, the working directory is not correct
# This corrects it, if necessary
base_dir = gsub(" ", "", paste(getwd(), '/identifying-users')) 
if(!(str_contains(getwd(), 'identifying-users'))) {
  setwd(base_dir)
}

ANALYSIS_TYPE = 'DAYS'

files_to_process = list("AllDays", "Weekdays") 
days_to_test = list("Day1", "Day2", "Day3", "Day4", "Day5", "Day6", "Day7")
days_map = c("Day1"="Monday", "Day2"="Tuesday", "Day3"="Wednesday", "Day4"="Thursday", "Day5"="Friday", "Day6"="Saturday", "Day7"="Sunday")

# TODO: All apps + time bins
# "TimeBehavProfAllApps.csv", "TimeBehavProf12hBins.csv")

# for reproducability
set.seed(2020)

print(" ")


if (ANALYSIS_TYPE == 'DAYS') {

  # testing the effect of days vs accuracy
  for(test_day in days_to_test) {

    day_str = paste("Test day:", days_map[test_day])

    # Get list of other days (i.e. days that are not test_day)
    other_days = list()
    for(day in days_to_test) {
      if (day != test_day)
      other_days <- append(other_days, day)
    }

    # Go from 1 day of training to 6 days of training, to be tested on test_day
    # Days are randomly chosen from other_days
    for(day_count in 1:6) {
      training_days = sample(other_days, day_count)

      # load data, train model
      file_loc = "csv_files\\AllDays.csv"
      RawData <- read.csv(file_loc, header = T)
      RawData$Person <- as.factor(as.character(RawData$Person))
      DurationsTrain <- select(filter(RawData, is.element(Day, training_days)), -Day)
      DurationsTest <- select(filter(RawData, Day == test_day), -Day) 

      # Train
      RForest <- randomForest::randomForest(Person ~ ., data = DurationsTrain, ntree = 3120, importance=TRUE) 

      # Test
      Preds <- predict(RForest, DurationsTest, type = 'class')
      confusionMatrix <- table(DurationsTest$Person, Preds)
      accuracy <- sum(diag(confusionMatrix))/sum(confusionMatrix) * 100
      accuracy = round(accuracy, 2)

      day_str = paste(day_str, "|| with", day_count, "day(s):", accuracy, "%")

    }

    print(day_str)

  } 


} else if (ANALYSIS_TYPE == 'ALLDAYS_WEEKDAY') {

  # testing weekdays-only vs all-days
  for(test_day in days_to_test) {

    day_str = paste("Test day:", days_map[test_day])
    day_output_count = 0

    for (f in files_to_process) {

      # Create file location
      file_loc = gsub(" ", "", paste("csv_files\\", f, '.csv')) 

      # Load in data from the file path
      RawData <- read.csv(file_loc, header = T)

      # Set person as the label of the data
      RawData$Person <- as.factor(as.character(RawData$Person))

      # Split into training and testing - tested per day
      DurationsTrain <- select(filter(RawData, Day != test_day), -Day) 
      DurationsTest <- select(filter(RawData, Day == test_day), -Day) 

      # Train
      RForest <- randomForest::randomForest(Person ~ ., data = DurationsTrain, ntree = 3120, importance=TRUE) 

      # Test
      Preds <- predict(RForest, DurationsTest, type = 'class')
      confusionMatrix <- table(DurationsTest$Person, Preds)
      accuracy <- sum(diag(confusionMatrix))/sum(confusionMatrix) * 100
      accuracy = round(accuracy, 2)

      if (!is.na(accuracy)) {
        day_str = paste(day_str, "|| the accuracy for", f, "is", accuracy, "%")
        day_output_count = day_output_count + 1
      }

    }  

    if (day_output_count > 0) {
      print(day_str)
    }

  }
}


