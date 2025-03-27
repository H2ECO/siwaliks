## 
## This script tunes and benchmarks modelling algorithms for dental ecometric-based climate reconstructions.
## This is performed within the mlr3 framework.
## There is another script for making predictions using Siwaliks fossil data (siwaliks_prediction_and_plottin.r)
## 
## If you just want to look at/have the predictions, you can skip this script.


## read in required packages (not all of them are probably required)
library(sf)
library(matrixStats)
library(colorspace)
library(terra)
library(data.table)
library(mlr3verse)
library(mlr3spatiotempcv)
library(mlr3learners)
library(mlr3viz)
library(paradox)
library(mlr3extralearners)
library(mlr3mbo)
library(bbotk)
library(gridExtra)
library(iml)

## Set the working directory
setwd("~/pCloudDrive/projects/siwaliks")


## Read in the training data (present natural with 4000m altitude cut-off)

## Read in the data. 
ModelledSubset <- fread("./model_inputs/subsetsubsetSpecies_div2_elev4000_Eurasia_PN.csv")

## Remove the easternmost tip of Siberia, which may cause troubles
## because of the coordinates 
ModelledSubset <- ModelledSubset[ModelledSubset$X > -20 & ModelledSubset$X < 179, ]


## Subset only required variables
modeldataBio12 <- ModelledSubset[, c("X", "Y", "BIO12_Mean", "HYP", "ALX", "BUN", "SF", "OT")]
modeldataBio01 <- ModelledSubset[, c("X", "Y", "BIO01_Mean", "HYP", "ALX", "BUN", "SF", "OT")]
## pairs(modeldataBio12[, 3:ncol(modeldataBio12)], lower.panel = NULL)
## pairs(modeldataBio01[, 3:ncol(modeldataBio01)], lower.panel = NULL)

## Let's leave SF out, if it is not reliable trait for these age ranges
dataBio12 <- modeldataBio12[, !"SF"]
dataBio01 <- modeldataBio01[, !"SF"]
#################################################################################################


## Define the "tasks" in mlr3 framework

## First, for Pann
taskBio12 = mlr3spatiotempcv::as_task_regr_st(
  dataBio12, 
  target = "BIO12_Mean", 
  id = "bio12_model",
  coordinate_names = c("X", "Y"),
  crs = "EPSG:4326",
  coords_as_features = FALSE
  )

taskBio12

## Tann
taskBio1 = mlr3spatiotempcv::as_task_regr_st(
  dataBio01, 
  target = "BIO01_Mean", 
  id = "bio1_model",
  coordinate_names = c("X", "Y"),
  crs = "EPSG:4326",
  coords_as_features = FALSE
  )

taskBio1


## Define the cross-validation scheme. In this case it is repeated (10 times) spatial five-fold cross-
## validation. Block size is 2900km, which is unnecessarily large, but let's use it anyway
tuningCVscheme = rsmp("repeated_spcv_block", range = rep(2900000L, 10), folds = 5, repeats = 10, 
selection = "random", hexagon = FALSE)

## Plotting different folds of repetition no. 1
autoplot(tuningCVscheme, task=taskBio1, repeats_id = 1)

########################################################
#
# Tann models
#
#######################################################3
## Next, we start to tune model hyperparameters.
## Fist Suppor Vector Machines

# Define the learner and set search space
learner_svm = lrn("regr.svm",
  type  = "eps-regression",
  kernel = "radial",
  predict_type = "response",
  cost = to_tune(0.01, 10),
  gamma = to_tune(0.01, 10)
)

## Set fallback learner in case of errors occur during cross-validations

learner_svm$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

# Create a data of parameter value combinations to evaluate.
designSVM = data.table(expand.grid(cost = c(0.01, 0.1, 1, 5, 10),
                                gamma = c(0.01, 0.1, 1, 5, 10)))


## Start parellisim with 10 workers (check how many cores your computer has!)
future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
svm_tuning_results = tune(
  tuner = tnr("design_points", design = designSVM),
  task = taskBio1,
  learner = learner_svm,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")

## Save tuning results
saveRDS(svm_tuning_results, file = "./model_outputs/svm_temp_tuning_results_itr50_s2000.rds")

## Read in, if you already have saved them
#svm_tuning_results <- readRDS("./svm_temp_tuning_results_itr50_s2000.rds")
#svm_tuning_results

## Check the best parameter value combinations
dat <- as.data.table(svm_tuning_results$archive)
head(dat[order(dat$regr.rmse, decreasing=FALSE), 1:5], 30)



## This would be the "best" (tuned) model
learner_svm_finT = lrn("regr.svm",
  type  = "eps-regression",
  kernel = "radial",
  predict_type = "response",
  cost = 0.1,
  gamma = 1
)

## Set fallback learner in case of errors
learner_svm_finT$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

###########################################################
## Next Random Forest

## Parameters to tune are mtry, min.node.size, and number of trees
designRF = data.table(expand.grid(mtry = c(1L, 2L, 3L),
                      min.node.size = c(1L, 5L, 10L, 15L),
                      num.trees = c(800L, 1000L, 1200L, 1500L))) ## 1, 5, 10))

# load learner and set search space
learner_rf = lrn("regr.ranger",
  predict_type = "response",
  mtry = to_tune(1L, 3L),
  min.node.size = to_tune(1L, 15L),
  num.trees = to_tune(800L, 1500L)
)


learner_rf$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
rf_tuning_results = tune(
  tuner = tnr("design_points", design = designRF),
  task = taskBio1,
  learner = learner_rf,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")


saveRDS(rf_tuning_results, file = "./model_outputs/rf_temp_tuning_results50_s2900.rds")

#rf_tuning_results <- readRDS("~/pCloudDrive/projects/siwaliks/rf_temp_tuning_results50_s2900.rds")
#rf_tuning_results

dat <- as.data.table(rf_tuning_results$archive)
head(dat[order(dat$regr.rmse, decreasing=FALSE), 1:5], 30)

## Tuned model
learner_rf_finT = lrn("regr.ranger",
  predict_type = "response",
  mtry = 2,
  min.node.size = 15,
  num.trees = 800
)

learner_rf_finT$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


########################333

## GBM

## This is tuned in piecewise manner. First number of threes and learning rate
## PART 1
designGBM1 = data.table(expand.grid(shrinkage = c(0.001, 0.01, 0.1, 0.5),
                      n.trees = c(500L, 1000L, 2000L, 3000L, 4000L, 6000L))) ## 1, 5, 10))


learner_gbm_1 = lrn("regr.gbm",
  predict_type = "response",
  shrinkage = to_tune(0.001, 0.5),
  n.trees = to_tune(500L, 6000L)
)

learner_gbm_1$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
gbm_tuning_results_1 = tune(
  tuner = tnr("design_points", design = designGBM1),
  task = taskBio1,
  learner = learner_gbm_1,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")


dat <- as.data.table(gbm_tuning_results_1$archive)

head(dat[order(dat$regr.rmse, decreasing=FALSE), 1:5], 30)


saveRDS(gbm_tuning_results_1, file = "./model_outputs/gbm_temp_tuning_1_results50_s2900.rds")

#gbm_tuning_results_1 <- readRDS("~/pCloudDrive/projects/siwaliks/gbm_temp_tuning_1_results50_s2900.rds")

#################################################

## Next interaction depth and minimum number of observations in the terminal nodes
## of the trees

### PART 2

designGBM2_2 = data.table(expand.grid(interaction.depth = c(4L, 5L, 6L, 7L, 8L),
n.minobsinnode = c(1L, 2L, 5L, 10L))) ## 1, 5, 10))


learner_gbm_2 = lrn("regr.gbm",
  predict_type = "response",
  shrinkage = 0.01,
  n.trees = 6000L,
  interaction.depth = to_tune(4L, 8L),
  n.minobsinnode = to_tune(1L, 10L)
)

learner_gbm_2$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
gbm_tuning_results_2 = tune(
  tuner = tnr("design_points", design = designGBM2_2),
  task = taskBio1,
  learner = learner_gbm_2,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")

dat2 <- as.data.table(gbm_tuning_results_2$archive)

head(dat2[order(dat2$regr.rmse, decreasing=FALSE), 1:5], 30)

saveRDS(gbm_tuning_results_2, file = "./model_outputs/gbm_temp_tuning_2_results50_s2900.rds")

gbm_tuning_results_2 <- readRDS("./model_outputs/gbm_temp_tuning_2_results50_s2900.rds")


### PART 3

## Re-tuning number of trees and learning rate
designGBM3 = data.table(expand.grid(shrinkage = c(0.001, 0.01, 0.1, 0.5),
                      n.trees = c(500L, 1000L, 2000L, 3000L, 4000L, 6000L))) ## 1, 5, 10))


learner_gbm_3 = lrn("regr.gbm",
  predict_type = "response",
  shrinkage = to_tune(0.001, 0.5),
  n.trees = to_tune(500L, 6000L),
  interaction.depth = 6,
  n.minobsinnode = 5

)

learner_gbm_3$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
gbm_tuning_results_3 = tune(
  tuner = tnr("design_points", design = designGBM3),
  task = taskBio1,
  learner = learner_gbm_3,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")

dat3 <- as.data.table(gbm_tuning_results_3$archive)

head(dat3[order(dat3$regr.rmse, decreasing=FALSE), 1:5], 30)

## Tuned model

learner_gbm_finT = lrn("regr.gbm",
  predict_type = "response",
  shrinkage = 0.01,
  n.trees = 4000L,
  interaction.depth = 6,
  n.minobsinnode = 5
  )

learner_gbm_finT$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


##########################################
## Xtreme gradient boosting is not used in the model ensemble this time.
## Tunign again in pieces

## XGB

## Number of trees and learning rate
designXGB1 = data.table(expand.grid(
                    nrounds = c(500L, 800L, 1000L, 2000L, 5000L),
                    eta = c(0.001, 0.01, 0.05, 0.1, 0.5) #, 
                    ##gamma = c(0, 5, 15),
                    ##max_depth = c(3L, 6L, 10L), 
                    ##min_child_weight = c(1, 3, 6)
              ))


## First stage learner, where only learning rate (eta)
## is tuned
# load learner and set search space
learner_xgb_1 = lrn("regr.xgboost",
  predict_type = "response",
  nrounds = to_tune(500L, 5000L),
  eta = to_tune(0.001, 0.5)
  #gamma = 0,
  #max_depth = 6,
  #colsample_bytree = 0.8,
  #min_child_weight = 1
  #subsample = 1
)

#learner_xgb_1$fallback = lrn("classif.featureless")
learner_xgb_1$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

tuningCVschemeXGB = rsmp("repeated_spcv_block", range = rep(2000000L, 10), folds = 5, repeats = 10, 
selection = "random", hexagon = FALSE)

future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
xgb_tuning_results_1 = tune(
  tuner = tnr("design_points", design = designXGB1),
  task = taskBio1,
  learner = learner_xgb_1,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")

saveRDS(xgb_tuning_results_1, file = "./model_outputs/xgb_temp_tuning_1_results50_s2900.rds")

dat2 <- as.data.table(xgb_tuning_results_1$archive)

head(dat2[order(dat2$regr.rmse, decreasing=FALSE), 1:5], 30)

## Part 2

## maximum depth and number of samples in a leaf
designXGB2 = data.table(expand.grid(
                    max_depth = c(4L, 5L, 6L, 7L, 8L), 
                    min_child_weight = c(1, 2, 5, 10)
              ))


## First stage learner, where only learning rate (eta)
## is tuned
# load learner and set search space
learner_xgb_2 = lrn("regr.xgboost",
  predict_type = "response",
  nrounds = 1000L,
  eta = 0.01,
  #gamma = 0,
  max_depth = to_tune(4L, 8L),
  #colsample_bytree = 0.8,
  min_child_weight = to_tune(1, 10)
  #subsample = 1
)

#learner_xgb_1$fallback = lrn("classif.featureless")
learner_xgb_2$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
xgb_tuning_results_2 = tune(
  tuner = tnr("design_points", design = designXGB2),
  task = taskBio1,
  learner = learner_xgb_2,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")

saveRDS(xgb_tuning_results_2, file = "./model_outputs/xgb_temp_tuning_2_results50_s2900.rds")


dat3 <- as.data.table(xgb_tuning_results_2$archive)

head(dat3[order(dat3$regr.rmse, decreasing=FALSE), 1:5], 30)


###
## PART 3

## Regularization parameters (not alpha, though)

designXGB3 = data.table(expand.grid(
                    gamma = c(0, 1, 5, 15),
                    lambda = c(0, 0.01, 0.1, 1, 10, 100)
                    #alpha = c(0, 0.01, 0.1, 1, 10, 100)
              ))


## First stage learner, where only learning rate (eta)
## is tuned
# load learner and set search space
learner_xgb_3 = lrn("regr.xgboost",
  predict_type = "response",
  nrounds = 1000L,
  eta = 0.01,
  max_depth = 6,
  min_child_weight = 10,
  gamma = to_tune(0, 15),
  lambda = to_tune(0, 100)
  #alpha = to_tune(0, 1e-2, 0.1, 1, 100, 1000, 10000)

)

learner_xgb_3$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
xgb_tuning_results_3 = tune(
  tuner = tnr("design_points", design = designXGB3),
  task = taskBio1,
  learner = learner_xgb_3,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")

saveRDS(xgb_tuning_results_3, file = "./model_outputs/xgb_temp_tuning_3_results50_s2900.rds")
xgb_tuning_results_3 <- readRDS("./model_outputs/xgb_temp_tuning_3_results50_s2900.rds")

dat3 <- as.data.table(xgb_tuning_results_3$archive)

head(dat3[order(dat3$regr.rmse, decreasing=FALSE), 1:5], 30)



## PART 4

## Re-tuning number of trees and learning rate
designXGB4 = data.table(expand.grid(
                    nrounds = c(800L, 1000L, 2000L, 5000L, 6000L),
                    eta = c(0.001, 0.01, 0.05, 0.1, 0.5) #, 
))

## First stage learner, where only learning rate (eta)
## is tuned
# load learner and set search space
learner_xgb_4 = lrn("regr.xgboost",
  predict_type = "response",
  nrounds = to_tune(800L, 6000L),
  eta = to_tune(0.001, 0.5),
  max_depth = 6,
  min_child_weight = 10,
  gamma = 0,
  lambda = 1
  #alpha = to_tune(0, 1e-2, 0.1, 1, 100, 1000, 10000)

)

#learner_xgb_1$fallback = lrn("classif.featureless")
learner_xgb_4$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
xgb_tuning_results_4 = tune(
  tuner = tnr("design_points", design = designXGB4),
  task = taskBio1,
  learner = learner_xgb_4,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")

dat4 <- as.data.table(xgb_tuning_results_4$archive)

head(dat4[order(dat4$regr.rmse, decreasing=FALSE), 1:5], 30)





## Tuned model
learner_xgb_finT = lrn("regr.xgboost",
  predict_type = "response",
  nrounds = 2000L,
  eta = 0.01,
  max_depth = 6,
  min_child_weight = 10,
  gamma = 0,
  lambda = 1
  #alpha = to_tune(0, 1e-2, 0.1, 1, 100, 1000, 10000)

)

learner_xgb_finT$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

###################################################################################################3

## MARS

## Two parameters to tune: level of interaction (degree) and 
## Maximum number of terms retained after pruning

designMARS = data.table(expand.grid(
                    degree = c(1, 2, 3),
                    nprune = c(5, 10, 15, 20, 25, 30, 35, 80)
              ))

## First stage learner, where only learning rate (eta)
## is tuned
# load learner and set search space
learner_mars = lrn("regr.earth",
  predict_type = "response",
  nprune = to_tune(5, 80),
  degree = to_tune(1, 3)
  
)

learner_mars$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))



future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
mars_tuning_results = tune(
  tuner = tnr("design_points", design = designMARS),
  task = taskBio1,
  learner = learner_mars,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")

saveRDS(mars_tuning_results, file = "./model_outputs/mars_temp_tuning_results50_s2900.rds")

dat <- as.data.table(mars_tuning_results$archive)

head(dat[order(dat$regr.rmse, decreasing=FALSE), 1:5], 30)


## Tuned model
learner_mars_finT = lrn("regr.earth",
  predict_type = "response",
  nprune = 20,
  degree = 2
  
)

learner_mars_finT$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

#########################################

## Scam model
## Read in the scam learner from a folder 
source("./R/learners/scam_learner.r")


## Define a learner. It's slightly different from normal as 
## scam is not part of mlr3 learners or extralearners
learner_scamT = LearnerRegrScam$new()

## Set the formula. Here you can specify the smooth constraints (bs)
learner_scamT$param_set$values$formula = BIO01_Mean ~ s(HYP, bs="mpd") + s(BUN, bs="mpi") + s(ALX, bs="mpd") + s(OT, bs="mpd")
learner_scamT$param_set$values$family = "gaussian"
learner_scamT$param_set$values$link = "identity"

learner_scamT$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


## GAM


## GAM
learner_gamT = lrn("regr.gam",
    method = "REML"  
)

## Set the formula. Here you can specify k values.
learner_gamT$param_set$values$formula = BIO01_Mean ~ s(HYP, k=5) + s(BUN, k=5) + s(ALX, k=5) + s(OT, k=5)

learner_gamT$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


 
#######################

## Benchmarking
# Next, benchmarking different methods

## Sligthly smaller block size here. Could be more reasonable to use this also in the tunings.
tuningCVscheme = rsmp("repeated_spcv_block", range = rep(2000000L, 10), folds = 5, repeats = 10, 
selection = "random", hexagon = FALSE)


tasks = taskBio1
## we can keep xgboost here
learners = c(learner_svm_finT, learner_rf_finT, learner_gbm_finT, learner_xgb_finT,
learner_mars_finT, learner_gamT, learner_scamT) 
rsmp_scheme = tuningCVscheme

design = benchmark_grid(tasks, learners, rsmp_scheme)
head(design)


## Run the benchmarking. This will take a lot of time.
future::plan("multisession", 
             workers = 10)
bmr = benchmark(design)
# stop parallelization
future:::ClusterRegistry("stop")

## print the results
bmr$aggregate(msr("regr.rmse"))

# Plot the results
##cairo_pdf("./bio1ModelComparison.pdf", width=8, height=5)
autoplot(bmr, measure = msr("regr.rmse"))
##dev.off()

###########################################################################################
### RESPONSE SHAPES
## You might want to have a look on what kind of effects different kinds of 
## ecometric variables have on Pann. Actually it would be more sensible to look
## the effects of climate on ecometric variables, but that is a topic of anoter study...

## This will use package "iml".
## It is possible to plot partial dependence plots and "ice" plots.
## The latter shows how the prediction in individual data point in the training data
## will change if the the values of focal variable are changed while other variables
## remain as they are.

## We will create a new "dataset"

## Predictos or features
bio1_x = taskBio1$data(cols = taskBio1$feature_names)
# target
bio1_y = taskBio1$data(cols = taskBio1$target_names)


## First, for the random forest
predictorRfT = Predictor$new(learner_rf_finT, data = bio1_x, y = bio1_y)

## One can also calculate variable importance. This, however, will take considerable amount of time.
##importance = FeatureImp$new(predictor, loss = "rmse", n.repetitions = 100)
##importance$plot()

## Next, estimate the effects of individual predictors. This will also take some time
## because it calculates also ice values
effBio1HYP = FeatureEffect$new(predictorRfT, feature = "HYP", method = "pdp+ice")
effBio1BUN = FeatureEffect$new(predictorRfT, feature = "BUN", method = "pdp+ice")
effBio1ALX = FeatureEffect$new(predictorRfT, feature = "ALX", method = "pdp+ice")
effBio1OT = FeatureEffect$new(predictorRfT, feature = "OT", method = "pdp+ice")

## Next, create plots
hypeffrfT <- effBio1HYP$plot()
buneffrfT <- effBio1BUN$plot()
alxeffrfT <- effBio1ALX$plot()
oteffrfT <- effBio1OT$plot()

## And plot them
grid.arrange(hypeffrfT, buneffrfT, alxeffrfT, oteffrfT, nrow = 2)

### The same for scam

predictorScamT = Predictor$new(learner_scamT, data = bio1_x, y = bio1_y)

#importanceScam = FeatureImp$new(predictorScam, loss = "rmse", n.repetitions = 100)
#importanceScam$plot()

effBio1HYPscam = FeatureEffect$new(predictorScamT, feature = "HYP", method = "pdp+ice")
effBio1BUNscam = FeatureEffect$new(predictorScamT, feature = "BUN", method = "pdp+ice")
effBio1ALXscam = FeatureEffect$new(predictorScamT, feature = "ALX", method = "pdp+ice")
effBio1OTscam = FeatureEffect$new(predictorScamT, feature = "OT", method = "pdp+ice")

hypeffscamT <- effBio1HYPscam$plot()
buneffscamT <- effBio1BUNscam$plot()
alxeffscamT <- effBio1ALXscam$plot()
oteffscamT <- effBio1OTscam$plot()

grid.arrange(hypeffscamT, buneffscamT, alxeffscamT, oteffscamT, nrow = 2)


#### SVM

predictorSvmT = Predictor$new(learner_svm_finT, data = bio1_x, y = bio1_y)
#importanceScam = FeatureImp$new(predictorScam, loss = "rmse", n.repetitions = 100)
#importanceScam$plot()

effBio1HYPsvm = FeatureEffect$new(predictorSvmT, feature = "HYP", method = "pdp+ice")
effBio1BUNsvm = FeatureEffect$new(predictorSvmT, feature = "BUN", method = "pdp+ice")
effBio1ALXsvm = FeatureEffect$new(predictorSvmT, feature = "ALX", method = "pdp+ice")
effBio1OTsvm = FeatureEffect$new(predictorSvmT, feature = "OT", method = "pdp+ice")

hypeffsvmT <- effBio1HYPsvm$plot()
buneffsvmT <- effBio1BUNsvm$plot()
alxeffsvmT <- effBio1ALXsvm$plot()
oteffsvmT <- effBio1OTsvm$plot()

grid.arrange(hypeffsvmT, buneffsvmT, alxeffsvmT, oteffsvmT, nrow = 2)


#### GBM
predictorGbmT = Predictor$new(learner_gbm_finT, data = bio1_x, y = bio1_y)

#importanceScam = FeatureImp$new(predictorScam, loss = "rmse", n.repetitions = 100)
#importanceScam$plot()

effBio1HYPgbm = FeatureEffect$new(predictorGbmT, feature = "HYP", method = "pdp+ice")
effBio1BUNgbm = FeatureEffect$new(predictorGbmT, feature = "BUN", method = "pdp+ice")
effBio1ALXgbm = FeatureEffect$new(predictorGbmT, feature = "ALX", method = "pdp+ice")
effBio1OTgbm = FeatureEffect$new(predictorGbmT, feature = "OT", method = "pdp+ice")

hypeffgbmT <- effBio1HYPgbm$plot()
buneffgbmT <- effBio1BUNgbm$plot()
alxeffgbmT <- effBio1ALXgbm$plot()
oteffgbmT <- effBio1OTgbm$plot()

grid.arrange(hypeffgbmT, buneffgbmT, alxeffgbmT, oteffgbmT, nrow = 2)


#### MARS

predictorMarsT = Predictor$new(learner_mars_finT, data = bio1_x, y = bio1_y)

#importanceScam = FeatureImp$new(predictorScam, loss = "rmse", n.repetitions = 100)
#importanceScam$plot()

effBio1HYPmars = FeatureEffect$new(predictorMarsT, feature = "HYP", method = "pdp+ice")
effBio1BUNmars = FeatureEffect$new(predictorMarsT, feature = "BUN", method = "pdp+ice")
effBio1ALXmars = FeatureEffect$new(predictorMarsT, feature = "ALX", method = "pdp+ice")
effBio1OTmars = FeatureEffect$new(predictorMarsT, feature = "OT", method = "pdp+ice")

hypeffmarsT <- effBio1HYPmars$plot()
buneffmarsT <- effBio1BUNmars$plot()
alxeffmarsT <- effBio1ALXmars$plot()
oteffmarsT <- effBio1OTmars$plot()

grid.arrange(hypeffmarsT, buneffmarsT, alxeffmarsT, oteffmarsT, nrow = 2)

#### GAM

predictorGamT = Predictor$new(learner_gamT, data = bio1_x, y = bio1_y)

#importanceScam = FeatureImp$new(predictorScam, loss = "rmse", n.repetitions = 100)
#importanceScam$plot()

effBio1HYPgam = FeatureEffect$new(predictorGamT, feature = "HYP", method = "pdp+ice")
effBio1BUNgam = FeatureEffect$new(predictorGamT, feature = "BUN", method = "pdp+ice")
effBio1ALXgam = FeatureEffect$new(predictorGamT, feature = "ALX", method = "pdp+ice")
effBio1OTgam = FeatureEffect$new(predictorGamT, feature = "OT", method = "pdp+ice")

hypeffgamT <- effBio1HYPgam$plot()
buneffgamT <- effBio1BUNgam$plot()
alxeffgamT <- effBio1ALXgam$plot()
oteffgamT <- effBio1OTgam$plot()

grid.arrange(hypeffgamT, buneffgamT, alxeffgamT, oteffgamT, nrow = 2)


##########################################################################################
##########################################
#########################################################################################33

## The same tunings for Pann

#######################################################3

## Fist Suppor Vector Machines

designSVM = data.table(expand.grid(cost = c(0.01, 0.1, 1, 5, 10),
                                gamma = c(0.01, 0.1, 1, 5, 10)))


# Define the learner and set search space
learner_svm = lrn("regr.svm",
  type  = "eps-regression",
  kernel = "radial",
  predict_type = "response",
  cost = to_tune(0.01, 10),
  gamma = to_tune(0.01, 10)
)

learner_svm$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


## Start parellisim with 10 workers
future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
svm_tuning_results = tune(
  tuner = tnr("design_points", design = designSVM),
  task = taskBio12,
  learner = learner_svm,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")


saveRDS(svm_tuning_results, file = "./model_outputs/svm_prec_tuning_results_itr50_s2000.rds")


#svm_tuning_results <- readRDS("./model_outputs/svm_prec_tuning_results_itr50_s2000.rds")

dat <- as.data.table(svm_tuning_results$archive)
head(dat[order(dat$regr.rmse, decreasing=FALSE), 1:5], 30)



## Tuned model
learner_svm_finP = lrn("regr.svm",
  type  = "eps-regression",
  kernel = "radial",
  predict_type = "response",
  cost = 0.1,
  gamma = 0.01
)

learner_svm_finP$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

#########################################################
## Random forest

## Parameters to tune are mtry, min.node.size, and number of trees
designRF = data.table(expand.grid(mtry = c(1L, 2L, 3L),
                      min.node.size = c(1L, 5L, 10L, 15L),
                      num.trees = c(800L, 1000L, 1200L, 1500L))) ## 1, 5, 10))

# load learner and set search space
learner_rf = lrn("regr.ranger",
  predict_type = "response",
  mtry = to_tune(1L, 3L),
  min.node.size = to_tune(1L, 15L),
  num.trees = to_tune(800L, 1500L)
)

learner_rf$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
rf_tuning_results = tune(
  tuner = tnr("design_points", design = designRF),
  task = taskBio12,
  learner = learner_rf,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")


saveRDS(rf_tuning_results, file = "./model_outputs/rf_prec_tuning_results50_s2900.rds")

##rf_tuning_results <- readRDS("./model_outputs/rf_prec_tuning_results50_s2900.rds")

dat <- as.data.table(rf_tuning_results$archive)
head(dat[order(dat$regr.rmse, decreasing=FALSE), 1:5], 30)

## Tuned model
learner_rf_finP = lrn("regr.ranger",
  predict_type = "response",
  mtry = 1,
  min.node.size = 15,
  num.trees = 1500
)

learner_rf_finP$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

########################333

## GBM and again in pieces (see above)
## PART 1
designGBM1 = data.table(expand.grid(shrinkage = c(0.001, 0.01, 0.1, 0.5),
                      n.trees = c(500L, 1000L, 2000L, 3000L, 4000L, 6000L))) ## 1, 5, 10))


learner_gbm_1 = lrn("regr.gbm",
  predict_type = "response",
  shrinkage = to_tune(0.001, 0.5),
  n.trees = to_tune(500L, 6000L)
)

learner_gbm_1$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
gbm_tuning_results_1 = tune(
  tuner = tnr("design_points", design = designGBM1),
  task = taskBio12,
  learner = learner_gbm_1,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")


saveRDS(gbm_tuning_results_1, file = "./model_outputs/gbm_prec_tuning_1_results50_s2900.rds")

#gbm_tuning_results_1 <- readRDS("./model_outputs/rf_prec_tuning_results50_s2900.rds")

dat <- as.data.table(gbm_tuning_results_1$archive)

head(dat[order(dat$regr.rmse, decreasing=FALSE), 1:5], 30)


#################################################

### PART 2

designGBM2_2 = data.table(expand.grid(interaction.depth = c(4L, 5L, 6L, 7L),
n.minobsinnode = c(10L, 15L, 20L))) ## 1, 5, 10))

learner_gbm_2 = lrn("regr.gbm",
  predict_type = "response",
  shrinkage = 0.1,
  n.trees = 1000L,
  interaction.depth = to_tune(4L, 8L),
  n.minobsinnode = to_tune(10L, 20L)
)


learner_gbm_2$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
gbm_tuning_results_2 = tune(
  tuner = tnr("design_points", design = designGBM2_2),
  task = taskBio12,
  learner = learner_gbm_2,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")

dat2 <- as.data.table(gbm_tuning_results_2$archive)

head(dat2[order(dat2$regr.rmse, decreasing=FALSE), 1:5], 30)

saveRDS(gbm_tuning_results_2, file = "./model_outputs/gbm_prec_tuning_2_2_results50_s2900.rds")


## PART 3
## re-tuning number of trees and learning rate

designGBM1 = data.table(expand.grid(shrinkage = c(0.001, 0.01, 0.1, 0.5),
                      n.trees = c(500L, 1000L, 2000L, 3000L, 4000L, 6000L))) ## 1, 5, 10))

learner_gbm_3 = lrn("regr.gbm",
  predict_type = "response",
  shrinkage = to_tune(0.001, 0.5),
  n.trees = to_tune(500L, 6000L),
  interaction.depth = 4,
  n.minobsinnode = 15
)


future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
gbm_tuning_results_3 = tune(
  tuner = tnr("design_points", design = designGBM1),
  task = taskBio12,
  learner = learner_gbm_3,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")

dat3 <- as.data.table(gbm_tuning_results_3$archive)

## Tuned model

learner_gbm_finP = lrn("regr.gbm",
  predict_type = "response",
  shrinkage = 0.01,
  n.trees = 1000L,
  interaction.depth = 4,
  n.minobsinnode = 15
  )

learner_gbm_finP$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


##########################################

## XGB

designXGB1 = data.table(expand.grid(
                    nrounds = c(500L, 800L, 1000L, 2000L, 5000L),
                    eta = c(0.001, 0.01, 0.05, 0.1, 0.5) #, 
                    
              ))


## First stage learner, where only learning rate (eta)
## is tuned
# load learner and set search space
learner_xgb_1 = lrn("regr.xgboost",
  predict_type = "response",
  nrounds = to_tune(500L, 5000L),
  eta = to_tune(0.001, 0.5)
)

#learner_xgb_1$fallback = lrn("classif.featureless")
learner_xgb_1$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
xgb_tuning_results_1 = tune(
  tuner = tnr("design_points", design = designXGB1),
  task = taskBio12,
  learner = learner_xgb_1,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")

saveRDS(xgb_tuning_results_1, file = "./xgb_prec_tuning_1_results50_s2900.rds")

dat <- as.data.table(xgb_tuning_results_1$archive)

head(dat[order(dat$regr.rmse, decreasing=FALSE), 1:5], 30)


## Part 2

designXGB2 = data.table(expand.grid(
                    max_depth = c(6L, 7L, 8L, 9L), 
                    min_child_weight = c(5, 10, 15, 20)
              ))


## First stage learner, where only learning rate (eta)
## is tuned
# load learner and set search space
learner_xgb_2 = lrn("regr.xgboost",
  predict_type = "response",
  nrounds = 2000L,
  eta = 0.001,
  max_depth = to_tune(6L, 9L),
  min_child_weight = to_tune(5, 20)
)

learner_xgb_2$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
xgb_tuning_results_2 = tune(
  tuner = tnr("design_points", design = designXGB2),
  task = taskBio12,
  learner = learner_xgb_2,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")

saveRDS(xgb_tuning_results_2, file = "./model_outputs/xgb_prec_tuning_2_results50_s2900.rds")

dat <- as.data.table(xgb_tuning_results_2$archive)

head(dat[order(dat$regr.rmse, decreasing=FALSE), 1:5], 30)

###
## PART 3

designXGB3 = data.table(expand.grid(
                    gamma = c(0, 1, 5, 15),
                    lambda = c(0, 0.01, 0.1, 1, 10, 100)
              ))


## First stage learner, where only learning rate (eta)
## is tuned
# load learner and set search space
learner_xgb_3 = lrn("regr.xgboost",
  predict_type = "response",
  nrounds = 2000L,
  eta = 0.001,
  max_depth = 7,
  min_child_weight = 10,
  gamma = to_tune(0, 15),
  lambda = to_tune(0, 100)
  #alpha = to_tune(0, 1e-2, 0.1, 1, 100, 1000, 10000)

)

learner_xgb_3$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
xgb_tuning_results_3 = tune(
  tuner = tnr("design_points", design = designXGB3),
  task = taskBio12,
  learner = learner_xgb_3,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")

saveRDS(xgb_tuning_results_3, file = "./model_outputs/xgb_prec_tuning_3_results50_s2900.rds")


dat <- as.data.table(xgb_tuning_results_3$archive)

head(dat[order(dat$regr.rmse, decreasing=FALSE), 1:5], 30)


## PART 4
## re-tuning number of trees and learning rate
designXGB1 = data.table(expand.grid(
                    nrounds = c(500L, 800L, 1000L, 2000L, 5000L),
                    eta = c(0.001, 0.01, 0.05, 0.1, 0.5) #, 
                    
              ))



learner_xgb_4 = lrn("regr.xgboost",
  predict_type = "response",
  nrounds = to_tune(500L, 5000L),
  eta = to_tune(0.001, 0.5),
  max_depth = 7,
  min_child_weight = 10,
  gamma = 0,
  lambda = 1

)


future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
xgb_tuning_results_4 = tune(
  tuner = tnr("design_points", design = designXGB1),
  task = taskBio12,
  learner = learner_xgb_4,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")


dat <- as.data.table(xgb_tuning_results_4$archive)

head(dat[order(dat$regr.rmse, decreasing=FALSE), 1:5], 30)

## Tuned model
learner_xgb_finP = lrn("regr.xgboost",
  predict_type = "response",
  nrounds = 5000L,
  eta = 0.001,
  max_depth = 7,
  min_child_weight = 10,
  gamma = 0,
  lambda = 1
  #alpha = to_tune(0, 1e-2, 0.1, 1, 100, 1000, 10000)

)

learner_xgb_finP$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))




###################################################################################################3
## MARS

designMARS = data.table(expand.grid(
                    degree = c(1, 2, 3),
                    nprune = c(5, 10, 15, 20, 25, 50)
              ))

## First stage learner, where only learning rate (eta)
## is tuned
# load learner and set search space
learner_mars = lrn("regr.earth",
  predict_type = "response",
  nprune = to_tune(5, 50),
  degree = to_tune(1, 3)
  
)

#learner_xgb_1$fallback = lrn("classif.featureless")
learner_mars$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))



future::plan("multisession", 
             workers = 10)
# run hyperparameter tuning
mars_tuning_results = tune(
  tuner = tnr("design_points", design = designMARS),
  task = taskBio12,
  learner = learner_mars,
  resampling = tuningCVscheme,
  measure = msr("regr.rmse")
)
# stop parallelization
future:::ClusterRegistry("stop")

saveRDS(mars_tuning_results, file = "./model_outputs/mars_prec_tuning_results50_s2900.rds")

dat <- as.data.table(mars_tuning_results$archive)

head(dat[order(dat$regr.rmse, decreasing=FALSE), 1:5], 30)


## Tuned model
learner_mars_finP = lrn("regr.earth",
  predict_type = "response",
  nprune = 10,
  degree = 3
  
)

learner_mars_finP$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

#########################################
## SCAM 

## Read in the scam learner from a folder 
source("./R/learners/scam_learner.r")

## Define a learner. It's slightly different from normal as 
## scam is not part of mlr3 learners or extralearners
learner_scamP = LearnerRegrScam$new()

## Set the formula
learner_scamP$param_set$values$formula = BIO12_Mean ~ s(HYP, bs="mpd") + s(BUN, bs="mpi") + s(ALX, bs="mpd") + s(OT, bs="mpd")
learner_scamP$param_set$values$family = "gaussian"
learner_scamP$param_set$values$link = "log"

learner_scamP$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


## GAM
learner_gamP = lrn("regr.gam",
    method = "REML"  
)

learner_gamP$param_set$values$formula = BIO12_Mean ~ s(HYP, k=5) + s(BUN, k=5) + s(ALX, k=5) + s(OT, k=5)

learner_gamP$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


#######################################################################################
### RESPONSE SHAPES
## You might want to have a look on what kind of effects different kinds of 
## ecometric variables have on Pann. Actually it would be more sensible to look
## the effects of climate on ecometric variables, but that is a topic of anoter study...

## This will use package "iml".
## It is possible to plot partial dependence plots and "ice" plots.
## The latter shows how the prediction in individual data point in the training data
## will change if the the values of focal variable are changed while other variables
## remain as they are.

## We will create a new "dataset"

## predictors (or features)
bio12_x = taskBio12$data(cols = taskBio12$feature_names)
# target
bio12_y = taskBio12$data(cols = taskBio12$target_names)

## Create a predictor object for random forest
predictorRfP = Predictor$new(learner_rf_finP, data = bio12_x, y = bio12_y)

## One can also calculate variable importance. This, however, will take considerable amount of time.
##importance = FeatureImp$new(predictor, loss = "rmse", n.repetitions = 100)
##importance$plot()

## Next, estimate the effects of individual predictors. This will also take some time
## because it calculates also ice values
effBio12HYP = FeatureEffect$new(predictorRfP, feature = "HYP", method = "pdp+ice")
effBio12BUN = FeatureEffect$new(predictorRfP, feature = "BUN", method = "pdp+ice")
effBio12ALX = FeatureEffect$new(predictorRfP, feature = "ALX", method = "pdp+ice")
effBio12OT = FeatureEffect$new(predictorRfP, feature = "OT", method = "pdp+ice")


## create plots
hypeffrf_P <- effBio12HYP$plot()
buneffrf_P <- effBio12BUN$plot()
alxeffrf_P <- effBio12ALX$plot()
oteffrf_P <- effBio12OT$plot()

## and plot them
grid.arrange(hypeffrf_P, buneffrf_P, alxeffrf_P, oteffrf_P, nrow = 2)

### The same for scam

predictorScamP = Predictor$new(learner_scamP, data = bio12_x, y = bio12_y)

#importanceScam = FeatureImp$new(predictorScamP, loss = "rmse", n.repetitions = 100)
#importanceScam$plot()

effBio12HYPscam = FeatureEffect$new(predictorScamP, feature = "HYP", method = "pdp+ice")
effBio12BUNscam = FeatureEffect$new(predictorScamP, feature = "BUN", method = "pdp+ice")
effBio12ALXscam = FeatureEffect$new(predictorScamP, feature = "ALX", method = "pdp+ice")
effBio12OTscam = FeatureEffect$new(predictorScamP, feature = "OT", method = "pdp+ice")

hypeffscamP <- effBio12HYPscam$plot()
buneffscamP <- effBio12BUNscam$plot()
alxeffscamP <- effBio12ALXscam$plot()
oteffscamP <- effBio12OTscam$plot()

grid.arrange(hypeffscamP, buneffscamP, alxeffscamP, oteffscamP, nrow = 2)


#### SVM
predictorSvmP = Predictor$new(learner_svm_finP, data = bio12_x, y = bio12_y)

#importanceScam = FeatureImp$new(predictorScamP, loss = "rmse", n.repetitions = 100)
#importanceScam$plot()

effBio12HYPsvm = FeatureEffect$new(predictorSvmP, feature = "HYP", method = "pdp+ice")
effBio12BUNsvm = FeatureEffect$new(predictorSvmP, feature = "BUN", method = "pdp+ice")
effBio12ALXsvm = FeatureEffect$new(predictorSvmP, feature = "ALX", method = "pdp+ice")
effBio12OTsvm = FeatureEffect$new(predictorSvmP, feature = "OT", method = "pdp+ice")

hypeffsvmP <- effBio12HYPsvm$plot()
buneffsvmP <- effBio12BUNsvm$plot()
alxeffsvmP <- effBio12ALXsvm$plot()
oteffsvmP <- effBio12OTsvm$plot()

grid.arrange(hypeffsvmP, buneffsvmP, alxeffsvmP, oteffsvmP, nrow = 2)

#### GBM
predictorGbmP = Predictor$new(learner_gbm_finP, data = bio12_x, y = bio12_y)

#importanceScam = FeatureImp$new(predictorScam, loss = "rmse", n.repetitions = 100)
#importanceScam$plot()

effBio12HYPgbm = FeatureEffect$new(predictorGbmP, feature = "HYP", method = "pdp+ice")
effBio12BUNgbm = FeatureEffect$new(predictorGbmP, feature = "BUN", method = "pdp+ice")
effBio12ALXgbm = FeatureEffect$new(predictorGbmP, feature = "ALX", method = "pdp+ice")
effBio12OTgbm = FeatureEffect$new(predictorGbmP, feature = "OT", method = "pdp+ice")

hypeffgbmP <- effBio12HYPgbm$plot()
buneffgbmP <- effBio12BUNgbm$plot()
alxeffgbmP <- effBio12ALXgbm$plot()
oteffgbmP <- effBio12OTgbm$plot()

grid.arrange(hypeffgbmP, buneffgbmP, alxeffgbmP, oteffgbmP, nrow = 2)


#### MARS

predictorMarsP = Predictor$new(learner_mars_finP, data = bio12_x, y = bio12_y)

#importanceScam = FeatureImp$new(predictorScam, loss = "rmse", n.repetitions = 100)
#importanceScam$plot()

effBio12HYPmars = FeatureEffect$new(predictorMarsP, feature = "HYP", method = "pdp+ice")
effBio12BUNmars = FeatureEffect$new(predictorMarsP, feature = "BUN", method = "pdp+ice")
effBio12ALXmars = FeatureEffect$new(predictorMarsP, feature = "ALX", method = "pdp+ice")
effBio12OTmars = FeatureEffect$new(predictorMarsP, feature = "OT", method = "pdp+ice")

hypeffmarsP <- effBio12HYPmars$plot()
buneffmarsP <- effBio12BUNmars$plot()
alxeffmarsP <- effBio12ALXmars$plot()
oteffmarsP <- effBio12OTmars$plot()

grid.arrange(hypeffmarsP, buneffmarsP, alxeffmarsP, oteffmarsP, nrow = 2)


#### GAM

predictorGamP = Predictor$new(learner_gamP, data = bio12_x, y = bio12_y)

#importanceScam = FeatureImp$new(predictorScam, loss = "rmse", n.repetitions = 100)
#importanceScam$plot()

effBio12HYPgam = FeatureEffect$new(predictorGamP, feature = "HYP", method = "pdp+ice")
effBio12BUNgam = FeatureEffect$new(predictorGamP, feature = "BUN", method = "pdp+ice")
effBio12ALXgam = FeatureEffect$new(predictorGamP, feature = "ALX", method = "pdp+ice")
effBio12OTgam = FeatureEffect$new(predictorGamP, feature = "OT", method = "pdp+ice")

hypeffgamP <- effBio12HYPgam$plot()
buneffgamP <- effBio12BUNgam$plot()
alxeffgamP <- effBio12ALXgam$plot()
oteffgamP <- effBio12OTgam$plot()

grid.arrange(hypeffgamP, buneffgamP, alxeffgamP, oteffgamP, nrow = 2)


######################################################################################## 
#######################
## Benchmarking
## Here we set up the benchmarking for both Tann and Pann models
## You need this if you want to compare the performace of different
## modelling algorithms. However, this will take some time.

tuningCVschemeB = rsmp("repeated_spcv_block", range = rep(2000000L, 10), folds = 5, repeats = 10, 
selection = "random", hexagon = FALSE)


tasksbio12 = taskBio12
learnersbio12 = c(learner_svm_finP, learner_rf_finP, learner_gbm_finP,
learner_mars_finP, learner_gamP, learner_scamP) 
rsmp_scheme = tuningCVschemeB

designbio12 = benchmark_grid(tasksbio12, learnersbio12, rsmp_scheme)



tasksbio1 = taskBio1
learnersbio1 = c(learner_svm_finT, learner_rf_finT, learner_gbm_finT,
learner_mars_finT, learner_gamT, learner_scamT) 

designbio1 = benchmark_grid(tasksbio1, learnersbio1, rsmp_scheme)

## For Pann models
future::plan("multisession", 
             workers = 10)
bmrbio12 = benchmark(designbio12)
# stop parallelization
future:::ClusterRegistry("stop")

bmrbio12$aggregate(msr("regr.rmse"))

cairo_pdf("./model_outputs/bio12ModelComparison.pdf", width=8, height=5)
autoplot(bmrbio12, measure = msr("regr.rmse"))
dev.off()


## For Tann models
future::plan("multisession", 
             workers = 10)
bmrbio1 = benchmark(designbio1)
# stop parallelization
future:::ClusterRegistry("stop")

bmrbio1$aggregate(msr("regr.rmse"))

cairo_pdf("./model_outputs/bio1ModelComparison.pdf", width=8, height=5)
autoplot(bmrbio1, measure = msr("regr.rmse"))
dev.off()


benchbio1 <- autoplot(bmrbio1, measure = msr("regr.rmse"))

benchbio12 <- autoplot(bmrbio12, measure = msr("regr.rmse"), title="Cross-validated performace of Pann models")


## Plotting them together
grid.arrange(benchbio1, benchbio12, ncol = 2)


####################################################################################################
