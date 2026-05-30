# Proportional-odds assumption checks for the ordered logistic model.

out_dir <- "Logistic_result/proportional_odds_assumption"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

required_packages <- c("readxl", "MASS", "ordinal", "openxlsx")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop("Missing required R packages: ", paste(missing_packages, collapse = ", "))
}

categorical_candidates <- c(
  "Gender", "Rural", "LeftBy", "Religion", "Married", "Level", "Education",
  "Exposure", "WorkTime", "SleepQuality", "Alcohol", "Smoke", "Exercise",
  "ChronicDisease", "Cluster"
)

data <- readxl::read_excel("MPLUS_result/abuse_pos_only_demographic.xlsx")
cluster_label <- read.table("MPLUS_result/dataLPA3.TXT", header = FALSE)
data$Cluster <- as.integer(cluster_label[[ncol(cluster_label)]])
data <- data[complete.cases(data), ]

categorical <- intersect(categorical_candidates, names(data))
feature_categorical <- setdiff(categorical, "Cluster")
continuous <- setdiff(names(data), categorical)

for (var in feature_categorical) {
  data[[var]] <- factor(data[[var]])
}

for (var in continuous) {
  if (is.numeric(data[[var]])) {
    data[[var]] <- as.numeric(scale(data[[var]]))
  }
}

data$RiskClass <- ordered(3 - as.integer(data$Cluster), levels = c(0, 1, 2))
predictors <- setdiff(names(data), c("Cluster", "RiskClass"))
model_formula <- as.formula(paste("RiskClass ~", paste(predictors, collapse = " + ")))
nominal_formula <- as.formula(paste("~", paste(predictors, collapse = " + ")))

fit_clm <- ordinal::clm(model_formula, data = data, link = "logit", Hess = TRUE)
nominal_test_result <- ordinal::nominal_test(fit_clm)

fit_clm_nominal <- ordinal::clm(
  model_formula,
  nominal = nominal_formula,
  data = data,
  link = "logit",
  Hess = TRUE
)
global_lr_test <- anova(fit_clm, fit_clm_nominal)

model_summary <- data.frame(
  Item = c(
    "N",
    "Outcome coding",
    "Model",
    "Main package",
    "Global proportional-odds test"
  ),
  Value = c(
    nrow(data),
    "RiskClass = 3 - Cluster; higher values indicate higher odds of the resource-constrained profile",
    paste(deparse(model_formula), collapse = " "),
    "ordinal::clm",
    "Likelihood-ratio comparison between proportional-odds and full nominal-effects cumulative logit models"
  )
)

brant_available <- requireNamespace("brant", quietly = TRUE)
brant_output <- data.frame(
  Package = "brant",
  Status = if (brant_available) "available and executed" else "not installed; skipped",
  Note = if (brant_available) "" else "Install brant to run the classic Brant test based on MASS::polr."
)

if (brant_available) {
  fit_polr <- MASS::polr(model_formula, data = data, method = "logistic", Hess = TRUE)
  brant_raw <- brant::brant(fit_polr)
  brant_capture <- capture.output(print(brant_raw))
  writeLines(brant_capture, file.path(out_dir, "brant_test_output.txt"))
  if (is.matrix(brant_raw) || is.data.frame(brant_raw)) {
    brant_output <- data.frame(Test = rownames(as.data.frame(brant_raw)), as.data.frame(brant_raw), row.names = NULL)
  }
}

wb <- openxlsx::createWorkbook()
openxlsx::addWorksheet(wb, "model_summary")
openxlsx::addWorksheet(wb, "nominal_test_by_term")
openxlsx::addWorksheet(wb, "global_nominal_LR_test")
openxlsx::addWorksheet(wb, "brant_status")

openxlsx::writeData(wb, "model_summary", model_summary)
openxlsx::writeData(wb, "nominal_test_by_term", data.frame(Term = rownames(as.data.frame(nominal_test_result)), as.data.frame(nominal_test_result), row.names = NULL))
openxlsx::writeData(wb, "global_nominal_LR_test", data.frame(Model = rownames(as.data.frame(global_lr_test)), as.data.frame(global_lr_test), row.names = NULL))
openxlsx::writeData(wb, "brant_status", brant_output)

output_path <- file.path(out_dir, "proportional_odds_assumption_tests.xlsx")
openxlsx::saveWorkbook(wb, output_path, overwrite = TRUE)

sink(file.path(out_dir, "proportional_odds_assumption_tests.txt"))
cat("Model formula:\n")
print(model_formula)
cat("\nordinal::nominal_test result:\n")
print(nominal_test_result)
cat("\nGlobal likelihood-ratio test comparing proportional and nominal-effects models:\n")
print(global_lr_test)
cat("\nbrant package status:\n")
print(brant_output)
sink()

cat("Saved proportional-odds assumption results to ", output_path, "\n", sep = "")
