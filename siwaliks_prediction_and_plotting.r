### PREDICTIONS
## This script produces Tann and Pann predictions to Siwaliks fossil data.class(
## These predictions are based on herbivore dental ecometrics. Modlellling algorithms are tuned and evaluated in another
## script (siwaliks_model_tuning.r). In this script, you will first read in the training data and fossil data.
## Then you fit the models to data and make predictions. Finally you will plot the results

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
library(segmented)
library(scam)
library(corrplot)




## Set the working directory
setwd("~/pCloudDrive/projects/siwaliks")

## Fossil data
siwadata <- read.csv("./model_inputs/fossils/dat_final.csv")

## Training data
ModelledSubset <- fread("./model_inputs/subsetsubsetSpecies_div2_elev4000_Eurasia_PN.csv")

## Remove the easternmost tip of Siberia, which may cause troubles
## because of the coordinates 
ModelledSubset <- ModelledSubset[ModelledSubset$X > -20 & ModelledSubset$X < 179, ]


## Subset only required variables
modeldataBio12 <- ModelledSubset[, c("X", "Y", "BIO12_Mean", "HYP", "ALX", "BUN", "SF", "OT")]
modeldataBio01 <- ModelledSubset[, c("X", "Y", "BIO01_Mean", "HYP", "ALX", "BUN", "SF", "OT")]

## Correlations between ecometric variables in the training data
Mbio12 <- cor(modeldataBio12[, 3:ncol(modeldataBio12)], method = "spearman")
Mbio01 <- cor(modeldataBio01[, 3:ncol(modeldataBio01)], method = "spearman")

## Let's leave SF out, if it is not reliable for these age ranges
dataBio12 <- modeldataBio12[, !"SF"]
dataBio01 <- modeldataBio01[, !"SF"]
#################################################################################################

## Define the modelling tasks (predicting Pann (bio12) and Tann (bio1))
taskBio12 = mlr3spatiotempcv::as_task_regr_st(
  dataBio12, 
  target = "BIO12_Mean", 
  id = "bio12_model",
  coordinate_names = c("X", "Y"),
  crs = "EPSG:4326",
  coords_as_features = FALSE
  )

taskBio12

## Define the task
taskBio1 = mlr3spatiotempcv::as_task_regr_st(
  dataBio01, 
  target = "BIO01_Mean", 
  id = "bio1_model",
  coordinate_names = c("X", "Y"),
  crs = "EPSG:4326",
  coords_as_features = FALSE
  )

taskBio1


########################################################
## Defining tuned learners (see another script for tuning)

## SVM

## Tuned Pann model
learner_svm_finP = lrn("regr.svm",
  type  = "eps-regression",
  kernel = "radial",
  predict_type = "response",
  cost = 0.1,
  gamma = 0.01
)

## Set fallback learner in case of errors
learner_svm_finP$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


## Tuned Tann model
learner_svm_finT = lrn("regr.svm",
  type  = "eps-regression",
  kernel = "radial",
  predict_type = "response",
  cost = 0.1,
  gamma = 1
)

## Set fallback learner in case of errors
learner_svm_finT$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


#####################
## RF

## Tuned Pann model
learner_rf_finP = lrn("regr.ranger",
  predict_type = "response",
  mtry = 1,
  min.node.size = 15,
  num.trees = 1500
)

learner_rf_finP$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

## Tuned Tann model
learner_rf_finT = lrn("regr.ranger",
  predict_type = "response",
  mtry = 2,
  min.node.size = 15,
  num.trees = 800
)

learner_rf_finT$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

#################
## GBM

## Tuned Pann model

learner_gbm_finP = lrn("regr.gbm",
  predict_type = "response",
  shrinkage = 0.01,
  n.trees = 1000L,
  interaction.depth = 4,
  n.minobsinnode = 15
  )

learner_gbm_finP$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

## Tuned Tann model

learner_gbm_finT = lrn("regr.gbm",
  predict_type = "response",
  shrinkage = 0.01,
  n.trees = 4000L,
  interaction.depth = 6,
  n.minobsinnode = 5
  )

learner_gbm_finT$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


###################
## XGB

## Tuned Pann model
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


## Tuned Tann model
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



#########################
## MARS

## Tuned Pann model
learner_mars_finP = lrn("regr.earth",
  predict_type = "response",
  nprune = 10,
  degree = 3
  
)

learner_mars_finP$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

## Tuned Tann model

learner_mars_finT = lrn("regr.earth",
  predict_type = "response",
  nprune = 20,
  degree = 2
  
)

learner_mars_finT$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))

##############################################################
## SCAM
## Read in the scam learner from a folder 
source("./R/learners/scam_learner.r")

## Define a learner. It's slightly different from normal as 
## scam is not part of mlr3 learners or extralearners

## Pann model
learner_scamP = LearnerRegrScam$new()
## Set the formula. Here, you can define shape constraints (bs)
learner_scamP$param_set$values$formula = BIO12_Mean ~ s(HYP, bs="mpd") + s(BUN, bs="mpi") + s(ALX, bs="mpd") + s(OT, bs="mpd")
learner_scamP$param_set$values$family = "gaussian"
learner_scamP$param_set$values$link = "log"
learner_scamP$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


## Tann model
learner_scamT = LearnerRegrScam$new()
## Set the formula
learner_scamT$param_set$values$formula = BIO01_Mean ~ s(HYP, bs="mpd") + s(BUN, bs="mpi") + s(ALX, bs="mpd") + s(OT, bs="mpd")
learner_scamT$param_set$values$family = "gaussian"
learner_scamT$param_set$values$link = "identity"
learner_scamT$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


#################################################
## GAM

## Pann model
learner_gamP = lrn("regr.gam",
    method = "REML"  
)
## Define the formula. Here you can set the k parameters
learner_gamP$param_set$values$formula = BIO12_Mean ~ s(HYP, k=5) + s(BUN, k=5) + s(ALX, k=5) + s(OT, k=5)
learner_gamP$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))


## Tann model
learner_gamT= lrn("regr.gam",
    method = "REML"  
)

learner_gamT$param_set$values$formula = BIO01_Mean ~ s(HYP, k=5) + s(BUN, k=5) + s(ALX, k=5) + s(OT, k=5)
learner_gamT$encapsulate(method="evaluate", fallback = lrn("regr.featureless"))



#####################################################################
#####################################################################

#######################

## Benchmarking
## Here we set up the benchmarking for both Tann and Pann models
## You need this if you want to include a plot showing the performance of different
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


## Running the benchmarking experiments.

## THIS WILL TAKE SOME TIME!

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


###################################################################
###################################################################
## Train the models using modern training data

## Pann models
learner_svm_finP$train(taskBio12)
learner_rf_finP$train(taskBio12)
learner_gbm_finP$train(taskBio12)
learner_xgb_finP$train(taskBio12)
learner_mars_finP$train(taskBio12)
learner_gamP$train(taskBio12)
learner_scamP$train(taskBio12)

## Tann models
learner_svm_finT$train(taskBio1)
learner_rf_finT$train(taskBio1)
learner_gbm_finT$train(taskBio1)
learner_xgb_finT$train(taskBio1)
learner_mars_finT$train(taskBio1)
learner_gamT$train(taskBio1)
learner_scamT$train(taskBio1)


#########################################################3
######################

### PREDICTING TO FOSSIL DATA

## Fossil data that only has relevant variables (predictors)
siwadata2 <- siwadata[, c("HYP", "BUN", "OT", "ALX")]


predSiwBio12RF = as.data.table(learner_rf_finP$predict_newdata(newdata = siwadata2, task = taskBio12))
predSiwBio12SCAM = as.data.table(learner_scamP$predict_newdata(newdata = siwadata2, task = taskBio12))
predSiwBio12SVM = as.data.table(learner_svm_finP$predict_newdata(newdata = siwadata2, task = taskBio12))
predSiwBio12GBM = as.data.table(learner_gbm_finP$predict_newdata(newdata = siwadata2, task = taskBio12))
predSiwBio12MARS = as.data.table(learner_mars_finP$predict_newdata(newdata = siwadata2, task = taskBio12))
predSiwBio12GAM = as.data.table(learner_gamP$predict_newdata(newdata = siwadata2, task = taskBio12))
predSiwBio12XGB = as.data.table(learner_xgb_finP$predict_newdata(newdata = siwadata2, task = taskBio12))


## Add predictos to the Siwaliks fossil data (original one with all the variables)
siwadata$precipPredRF <- predSiwBio12RF$response
siwadata$precipPredSCAM <- predSiwBio12SCAM$response
siwadata$precipPredSVM <- predSiwBio12SVM$response
siwadata$precipPredGBM <- predSiwBio12GBM$response
siwadata$precipPredMARS <- predSiwBio12MARS$response
siwadata$precipPredGAM <- predSiwBio12GAM$response
siwadata$precipPredXGB <- predSiwBio12XGB$response

## Calculate the ensemble as a median of relevant model predictions
siwadata$ensemblePrec <- rowMeans(siwadata[, 14:19]) ## Let's leave XGB out


## The same for Tann
### TANN


predSiwBio1RF = as.data.table(learner_rf_finT$predict_newdata(newdata = siwadata2, task = taskBio1))
predSiwBio1SCAM = as.data.table(learner_scamT$predict_newdata(newdata = siwadata2, task = taskBio1))
predSiwBio1SVM = as.data.table(learner_svm_finT$predict_newdata(newdata = siwadata2, task = taskBio1))
predSiwBio1GBM = as.data.table(learner_gbm_finT$predict_newdata(newdata = siwadata2, task = taskBio1))
predSiwBio1MARS = as.data.table(learner_mars_finT$predict_newdata(newdata = siwadata2, task = taskBio1))
predSiwBio1GAM = as.data.table(learner_gamT$predict_newdata(newdata = siwadata2, task = taskBio1))
predSiwBio1XGB = as.data.table(learner_xgb_finT$predict_newdata(newdata = siwadata2, task = taskBio1))

## Add predictions to the data
siwadata$tempPredRF <- predSiwBio1RF$response
siwadata$tempPredSCAM <- predSiwBio1SCAM$response
siwadata$tempPredSVM <- predSiwBio1SVM$response
siwadata$tempPredGBM <- predSiwBio1GBM$response
siwadata$tempPredMARS <-  predSiwBio1MARS$response
siwadata$tempPredGAM <- predSiwBio1GAM$response
siwadata$tempPredXGB <- predSiwBio1XGB$response

names(siwadata)
siwadata$ensembleTemp <- rowMeans(siwadata[, 22:27]) ## XGB out



##################################################################
## BREAK-POINT ANALYSIS
## Estimate break-points in the trend using "segmented" package
# Detections of break points

lmTemp <- lm(ensembleTemp~AVERAGE.AGE, data=siwadata)
seg_Temp <- segmented(lmTemp, npsi=2)
summary(seg_Temp)

lmPrec <- lm(ensemblePrec~AVERAGE.AGE, data=siwadata)
seg_Prec <- segmented(lmPrec, npsi=2)
summary(seg_Prec)


#######################################################

## Hypsodonty-only precipitation model
## The most simplest (and maybe most robust) precipitation model
## This is based on scam
dataBio12HYP <- dataBio12[, c("BIO12_Mean", "HYP")]
scamM <- scam(BIO12_Mean ~ s(HYP, bs="mpd"), data=dataBio12HYP)

dataBio12HYP$Preds <- scamM$fitted.values

siwadata$precipHYP <- predict(scamM, newdata=siwadata)



#######################################################

#### PLOTTING

## The pdf file contains all the different plots within a single document. You can
## of course just run different plots separately


cairo_pdf("./model_outputs/siwaliksModels2.pdf", width=9, height=8) # dont run this unles you want to make a single document for all plots

## This plots temporal change in ecometric variables in Siwaliks

par(mfrow=c(3,2), mar=c(4,4,3,1))
## Hypsodonty
with(siwadata, plot(AVERAGE.AGE*-1, HYP, xaxt="n", xlab="Age (Ma)", las=1))
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))
text(-1.9, 2.45, "Change through time in dental ecometrics in Siwaliks", xpd=NA, cex=1.5)

## Bunodonty
with(siwadata, plot(AVERAGE.AGE*-1, BUN, xaxt="n", xlab="Age (Ma)", las=1))
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))

## Acute lophs
with(siwadata, plot(AVERAGE.AGE*-1, ALX, xaxt="n", xlab="Age (Ma)", las=1))
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))

## Structural fortification
with(siwadata, plot(AVERAGE.AGE*-1, SF, xaxt="n", xlab="Age (Ma)", las=1))
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))
## Interestingly SF is not completely zero before 10 Ma

## Occlusal topography
with(siwadata, plot(AVERAGE.AGE*-1, OT, xaxt="n", xlab="Age (Ma)", las=1))
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))

## Obtuse lophs
with(siwadata, plot(AVERAGE.AGE*-1, OL, xaxt="n", xlab="Age (Ma)", las=1))
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))

###################
## This plots relationships between ecometric variables in the fossil data
pairs(siwadata[, c("HYP", "BUN", "ALX", "SF", "OT", "OL")], main="Relationships between ecometric variables in the fossil data")

##################
## This plots correlations between ecometric variables and climate variables in the training data
layout.matrix <- matrix(c(1, 3, 2, 3), nrow = 2, ncol = 2)
layout(mat = layout.matrix)
corrplot(Mbio12, method="number", title="Present natural Eurasia correlations\n Pann and ecometrics",
mar=c(0,0,3,0))

## This and relationship between Tann and Pann in Eurasian modern training data
corrplot(Mbio01, method="number", title="Present natural Eurasia correlations\n Tann and ecometrics",
mar=c(0,0,3,0))
par(mar=c(5,5,3,3))
plot(ModelledSubset$BIO01_Mean, ModelledSubset$BIO12_Mean,
xlab="Tann (C) in Eurasia", 
ylab="Pann (mm) in Eurasia", main="Relationship between Tann and Pann in modern Eurasia")
text(0, 5000, paste("r = ", 
round(cor(ModelledSubset$BIO01_Mean, ModelledSubset$BIO12_Mean), 3)),
cex=1.5)

###################
## This plots the results of model benchmarking experiments
## Make sure that you have run benchmarks, if you want to plot this (see above)
grid.arrange(benchbio1, benchbio12, ncol = 2)

###################
## This plots Tann predictions by different models over time in Siwaliks 

par(mfrow=c(3,2), mar=c(4,4,3,1))
with(siwadata, plot(AVERAGE.AGE*-1, tempPredRF, xaxt="n", 
xlab="Age (Ma)", ylab="Tann (C)", las=1))
text(-17, 5, "random forest")
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))
text(-1.9, 29.5, "Modelled Tann (C) in Siwaliks", xpd=NA, cex=1.5)

with(siwadata, plot(AVERAGE.AGE*-1, tempPredSCAM, xaxt="n", 
xlab="Age (Ma)", ylab="Tann (C)", las=1))
text(-17, 5, "shape constrained additive models")
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))

with(siwadata, plot(AVERAGE.AGE*-1, tempPredSVM, xaxt="n", 
xlab="Age (Ma)", ylab="Tann (C)", las=1))
text(-17, 5, "support vector machines")
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))

with(siwadata, plot(AVERAGE.AGE*-1, tempPredGBM, xaxt="n", 
xlab="Age (Ma)", ylab="Tann (C)", las=1), ylab="Tann (C)", las=1)
text(-17, 5, "gradient boosting machines")
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))

with(siwadata, plot(AVERAGE.AGE*-1, tempPredMARS, xaxt="n", 
xlab="Age (Ma)", ylab="Tann (C)", las=1))
text(-17, 5, "multivariate adaptive regression splines")
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))

with(siwadata, plot(AVERAGE.AGE*-1, tempPredGAM, xaxt="n", 
xlab="Age (Ma)", ylab="Tann (C)", las=1))
text(- 17, 5, "generalized additive models")
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))

#######################
## This plots crelationships between different predictions in the Siwaliks data
pairs(siwadata[, 22:27], main="Correlations between Tann predictions")


########################
## This plots different Pann predictions over time in the Siwaliks data
par(mfrow=c(3,2), mar=c(4,4,3,1))
with(siwadata, plot(AVERAGE.AGE*-1, precipPredRF, xaxt="n", 
xlab="Age (Ma)", ylab="Pann (mm)", las=1))
text(-17, 500, "random forest")
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))
text(-1.9, 3500, "Modelled Pann (mm) in Siwaliks", xpd=NA, cex=1.5)

with(siwadata, plot(AVERAGE.AGE*-1, precipPredSCAM, xaxt="n", 
xlab="Age (Ma)", ylab="Pann (mm)", las=1))
text(-17, 500, "shape constrained additive models")
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))

with(siwadata, plot(AVERAGE.AGE*-1, precipPredSVM, xaxt="n", 
xlab="Age (Ma)", ylab="Pann (mm)", las=1))
text(-17, 500, "support vector machines")
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))

with(siwadata, plot(AVERAGE.AGE*-1, precipPredGBM, xaxt="n", 
xlab="Age (Ma)", ylab="Pann (mm)", las=1), ylab="Tann (C)", las=1)
text(-17, 500, "gradient boosting machines")
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))

with(siwadata, plot(AVERAGE.AGE*-1, precipPredMARS, xaxt="n", 
xlab="Age (Ma)", ylab="Pann (mm)", las=1))
text(-17, 500, "multivariate adaptive regression splines")
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))

with(siwadata, plot(AVERAGE.AGE*-1, precipPredGAM, xaxt="n", 
xlab="Age (Ma)", ylab="Pann (mm)", las=1))
text(- 17, 500, "generalized additive models")
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))

######################################
## This plots relationships between different Pann predictions
pairs(siwadata[, 14:19], main="Correlations between Pann predictions")


############################
## This plots ensemble predictions for Tann and Pann
par(mfrow=c(2,1), mar=c(4,6,1,1))
with(siwadata, plot(AVERAGE.AGE, ensembleTemp, xaxt="n",
xlab="Age (Ma)", ylab="Tann (C)", las=1))
lines(lowess(siwadata$AVERAGE.AGE, siwadata$ensembleTemp, f=0.5))
plot(seg_Temp, add=TRUE)
lines(seg_Temp)
axis(1, at=seq(3, 22, 1))
text(17, 2.5, "Ensemble prediction Tann (C)")

with(siwadata, plot(AVERAGE.AGE, ensemblePrec, xaxt="n",
xlab="Age (Ma)", ylab="Pann (mm)", las=1))
lines(lowess(siwadata$AVERAGE.AGE, siwadata$ensemblePrec, f=0.5))
plot(seg_Prec, add=TRUE)
lines(seg_Prec)
axis(1, at=seq(3, 22, 1))
text(17, 700, "Ensemble prediction Pann (mm)")

###########################
## This plots the relationship between hypsodonty and Pann in the modern training data
## as well as the predicted precipitation based onl on hypsodonty. 
## The prediction is based on scam model
par(mfrow=c(2,1), mar=c(4,5,3,1))
plot(dataBio12$HYP, dataBio12$BIO12_Mean, xlab="HYP", ylab="Pann (mm)",
main="SCAM fit (Pann ~ HYP) in modern data (present natural)")
with(dataBio12HYP[order(dataBio12HYP$HYP),],lines(Preds~HYP, col="red", lwd=2))

with(siwadata, plot(AVERAGE.AGE*-1, precipHYP, xaxt="n", 
xlab="Age (Ma)", ylab="Pann (mm)", las=1, main="Predicted precipitation"))
text(- 17, 500, "scam based only on HYP")
#lines(lowess(siwadata$AVERAGE.AGE*-1, siwadata$tempPredGAM, f=0.5))
axis(1, at=seq(-20,-5, 5), lab=seq(20, 5, -5))



dev.off() ## don't run this unless you want to save tall these plots in to single file (see above).

