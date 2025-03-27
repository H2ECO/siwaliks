
#################################################
## SCAM learner for mlr3

library(mlr3)
library(paradox)
library(R6)
library(mlr3misc)
library(Hmisc)



LearnerRegrScam = R6Class("LearnerRegrScam",
  inherit = LearnerRegr,

  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps = ps(
        family = p_fct(
          levels = c("gaussian", "poisson", "quasipoisson", "Gamma", "inverse.gaussian"),
          tags = "train"),
        formula = p_uty(tags = "train"),
        offset = p_uty(default = NULL, tags = "train"),
        ## link is new
        link = p_fct(
          levels = c(
            "logit", "probit", "cauchit", "cloglog", "identity",
            "log", "sqrt", "1/mu^2", "inverse"), tags = "train"),
        optimizer = p_uty(default = c("bfgs","newton"), tags = "train"),
        optim.method = p_uty(default = c("Nelder-Mead","fd"), tags = "train"),
        scale = p_dbl(default = 0, tags = "train"),
        knots = p_uty(default = NULL, tags = "train"),
        sp = p_uty(default = NULL, tags = "train"),
        gamma = p_dbl(default = 1, lower = 1, tags = "train"),
        drop.unused.levels = p_lgl(default = TRUE, tags = "train"),
        block.size = p_int(default = 1000L, tags = "predict"),
        unconditional = p_lgl(default = FALSE, tags = "predict")
      )
      ps$values = list(family = "gaussian", link = "log")

      super$initialize(
        id = "regr.scam",
        packages = c("mlr3extralearners", "scam"),
        feature_types = c("logical", "integer", "numeric", "factor"),
        predict_types = c("response", "se"),
        param_set = ps,
        properties = "weights",
        man = "mlr3extralearners::mlr_learners_regr.scam",
        label = "Shape Constrained Additive Regression Model"
      )
    }
  ),

  private = list(
    .train = function(task) {
      #pars = learner_scam_T$param_set$get_values(tags = "train")
      pars = self$param_set$get_values(tags = "train")
      ## new below
      family_args = pars[names(pars) == "link"]
      pars$link = NULL
      #
      control_pars = pars[names(pars) %in% formalArgs(scam::scam.control)]
      pars = pars[!(names(pars) %in% names(control_pars))]
      #pars = pars[names(pars) %nin% formalArgs(scam::scam.control)]
      
       # add family to parameters, this is new
      family_fn = getFromNamespace(pars$family, ns = "stats")
      pars$family = do.call(family_fn, args = family_args)
      #pars$family = invoke(family_fn, .args = family_args)
      #
      data = task$data(cols = c(task$feature_names, task$target_names))
      if ("weights" %in% task$properties) {
        pars = insert_named(pars, list(weights = task$weights$weight))
      }

      if (is.null(pars$formula)) {
        formula = stats::as.formula(paste(
          task$target_names,
          "~",
          paste(task$feature_names, collapse = " + ")
        ))
        pars$formula = formula
      }

      if (length(control_pars)) {
        control_obj = do.call(scam::scam.control, args = control_pars)
        ##invoke(scam::scam.control, .args = control_pars)
        pars = pars[!(names(pars) %in% names(control_pars))]
      } else {
        control_obj = scam::scam.control()
      }

      rlang::invoke(
        scam::scam,
        data = data,
        .args = pars,
        control = control_obj
      )
    },

    .predict = function(task) {
      # get parameters with tag "predict"

      pars = self$param_set$get_values(tags = "predict")

      # get newdata and ensure same ordering in train and predict
      newdata = mlr3extralearners:::ordered_features(task, self)

      include_se = (self$predict_type == "se")

      preds = rlang::invoke(
        predict,
        self$model,
        newdata = newdata,
        type = "response",
        newdata.guaranteed = TRUE,
        se.fit = include_se,
        .args = pars
      )

      if (include_se) {
        list(response = preds$fit, se = preds$se)
      } else {
        list(response = preds)
      }
    }
  )
)
