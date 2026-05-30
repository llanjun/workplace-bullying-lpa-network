library(bootnet)
library(networktools)
library(NetworkComparisonTest)
library(openxlsx)
library(qgraph)
library(ggplot2)
library(reshape2)
library(dplyr)
library(cowplot)
library(gridExtra)

BEST_CLASS <- 3
N_BOOT <- 1000
N_CORES <- min(parallel::detectCores(), 4)
FORCE_RERUN <- TRUE
BRIDGE_CI_VERSION <- "bridge_ci_by_boot_name_v2"

# Color palette for 3-class comparison
CLASS_COLORS <- c("Class1" = "#E41A1C", "Class2" = "#4DAF4A", "Class3" = "#377EB8")

skip_if_done <- function(rds_path, expr) {
  if (file.exists(rds_path) && !FORCE_RERUN) {
    message("    Loading existing: ", basename(rds_path))
    readRDS(rds_path)
  } else {
    force(expr)
  }
}

file_info_row <- function(path) {
  if (!file.exists(path)) {
    return(data.frame(
      file = path, exists = FALSE, size_bytes = NA_real_, modified = NA_character_
    ))
  }
  info <- file.info(path)
  data.frame(
    file = path,
    exists = TRUE,
    size_bytes = info$size,
    modified = format(info$mtime, "%Y-%m-%d %H:%M:%S")
  )
}

write_run_manifest <- function(output_dir, analysis_name) {
  input_files <- c(
    file.path("MPLUS_result", "abuse_pos_only_PID_PSR_ADP.xlsx"),
    file.path("MPLUS_result", sprintf("lpa_classification_%dclass.xlsx", BEST_CLASS)),
    file.path("Logistic_result", "significant_controls_with_cluster.csv"),
    file.path("Logistic_result", "lpa_items_complete_case.csv")
  )
  manifest <- do.call(rbind, lapply(input_files, file_info_row))
  manifest$analysis <- analysis_name
  manifest$best_class <- BEST_CLASS
  manifest$n_boot <- N_BOOT
  manifest$n_cores <- N_CORES
  manifest$force_rerun <- FORCE_RERUN
  manifest$bridge_ci_version <- BRIDGE_CI_VERSION
  write.csv(manifest, file.path(output_dir, "run_manifest.csv"), row.names = FALSE)
}

message("Using ", N_CORES, " CPU core(s) for bootstrap (FORCE_RERUN = ", FORCE_RERUN, ")")

for (analysis_name in c("base_continuous", "controlled_continuous")) {
  message("Running ", analysis_name)

  # ── Setup ──────────────────────────────────────────────────
  output_dir <- file.path("Network_result", analysis_name)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  write_run_manifest(output_dir, analysis_name)

  # ── Assumptions ────────────────────────────────────────────
  if (analysis_name == "base_continuous") {
    text <- c(
      "Base continuous network analysis.",
      "PID, PSR, and ADP items are 1-5 Likert responses and are treated as approximately continuous variables.",
      "This is consistent with the continuous-indicator LPA model.",
      "The network model is EBICglasso using the default correlation handling in bootnet.",
      "Missing data are handled by listwise deletion via bootnet default cor = 'auto'.",
      "Network accuracy is evaluated with nonparametric bootstrap confidence intervals for edge weights.",
      "Centrality stability is evaluated with case-dropping bootstrap and reported as CS coefficients.",
      "Bridge centrality stability is evaluated with case-dropping bootstrap for bridge strength and bridge expected influence."
    )
  } else {
    text <- c(
      "Controlled continuous network analysis.",
      "Potential confounders are taken from Logistic_result/significant_controls_with_cluster.csv.",
      "Each PID, PSR, and ADP item is residualized using linear regression before network estimation.",
      "The residualized variables are treated as continuous residual scores and estimated with EBICglasso.",
      "Missing data are handled by listwise deletion via bootnet default cor = 'auto' in both stages.",
      "Network accuracy is evaluated with nonparametric bootstrap confidence intervals for edge weights.",
      "Centrality stability is evaluated with case-dropping bootstrap and reported as CS coefficients.",
      "Bridge centrality stability is evaluated with case-dropping bootstrap for bridge strength and bridge expected influence."
    )
  }
  writeLines(text, file.path(output_dir, "network_assumptions.txt"))

  # ── Read LPA data ──────────────────────────────────────────
  df <- read.xlsx(file.path("MPLUS_result", "abuse_pos_only_PID_PSR_ADP.xlsx"), sheet = 1)
  clusters <- read.xlsx(
    file.path("MPLUS_result", sprintf("lpa_classification_%dclass.xlsx", BEST_CLASS)),
    sheet = 1
  )
  df$cluster <- as.factor(clusters$Cluster)

  # ── Residualize (controlled only) ──────────────────────────
  if (analysis_name == "controlled_continuous") {
    control_file <- file.path("Logistic_result", "significant_controls_with_cluster.csv")
    if (!file.exists(control_file)) {
      control_file <- file.path("..", "Logistic_result", "significant_controls_with_cluster.csv")
    }
    control_with_label <- read.csv(control_file)
    control_vars <- setdiff(colnames(control_with_label), "Cluster")

    if (length(control_vars) > 0) {
      aligned_file <- file.path("Logistic_result", "lpa_items_complete_case.csv")
      if (!file.exists(aligned_file)) {
        aligned_file <- file.path("..", "Logistic_result", "lpa_items_complete_case.csv")
      }
      ob_data <- read.csv(aligned_file)
      stopifnot(nrow(ob_data) == nrow(df),
                all(control_with_label$Cluster == clusters$Cluster))

      total_data <- data.frame(ob_data, control_with_label[, control_vars, drop = FALSE])
      residuals_data <- as.data.frame(matrix(nrow = nrow(ob_data), ncol = ncol(ob_data)))
      colnames(residuals_data) <- colnames(ob_data)
      for (i in seq_along(ob_data)) {
        residuals_data[, i] <- residuals(
          lm(reformulate(control_vars, response = colnames(ob_data)[i]), data = total_data,
           na.action = na.exclude)
        )
      }
      residuals_data$cluster <- as.factor(control_with_label$Cluster)
      df <- residuals_data
    } else {
      message("No significant controls found; controlled analysis uses unadjusted item data.")
    }
  }

  # ── Communities ────────────────────────────────────────────
  communities <- c(rep("Professional Identity, PID", 9),
                   rep("Psychological Resilience, PSR", 14),
                   rep("Adaptive Performance, ADP", 13))
  stopifnot(length(communities) == ncol(df) - 1)

  # ── Per-cluster network estimation ─────────────────────────
  networks <- list()
  plots <- list()
  bridges <- list()
  boot_cases <- list()
  layout <- "spring"

  for (k in sort(unique(as.integer(as.character(df$cluster))))) {
    message("  Cluster ", k)
    subset_data <- df[df$cluster == k, colnames(df) != "cluster", drop = FALSE]
    if (nrow(subset_data) < ncol(subset_data)) {
      warning("Cluster ", k, " has fewer observations (", nrow(subset_data),
              ") than variables (", ncol(subset_data), "); EBICglasso may be unstable")
    }

    set.seed(20260422)
    net <- estimateNetwork(subset_data, default = "EBICglasso")
    networks[[as.character(k)]] <- net

    pdf(file.path(output_dir, sprintf("cluster%d_network.pdf", k)))
    plot_net <- plot(net, layout = layout, title = paste("Cluster", k), negDashed = TRUE)
    dev.off()
    plots[[as.character(k)]] <- plot_net
    if (k == 1) layout <- plot_net$layout

    pdf(file.path(output_dir, sprintf("cluster%d_centrality.pdf", k)))
    centralityPlot(plot_net, scale = "relative", orderBy = "ExpectedInfluence",
                   include = c("ExpectedInfluence", "Strength"))
    dev.off()

    bridge_result <- bridge(plot_net, communities = communities)
    bridges[[as.character(k)]] <- bridge_result
    write.csv(
      as.data.frame(bridge_result[c("Bridge Strength", "Bridge Expected Influence (1-step)")]),
      file.path(output_dir, sprintf("cluster%d_bridge_centrality.csv", k))
    )

    edge_rds <- file.path(output_dir, sprintf("cluster%d_edge_bootstrap.rds", k))
    boot_edge <- skip_if_done(edge_rds, {
      set.seed(20260423)
      bootnet(net, nBoots = N_BOOT, nCores = N_CORES, type = "nonparametric",
              statistics = c("edge", "strength", "expectedInfluence"))
    })
    if (!file.exists(edge_rds) || FORCE_RERUN) saveRDS(boot_edge, edge_rds)

    # ── Bridge centrality bootstrap CI ────────────────────────
    bridge_ci_csv <- file.path(output_dir, sprintf("cluster%d_bridge_boot_CI.csv", k))
    bridge_ci_meta <- file.path(output_dir, sprintf("cluster%d_bridge_boot_CI.meta.txt", k))
    bridge_ci_is_current <- file.exists(bridge_ci_csv) &&
      file.exists(bridge_ci_meta) &&
      identical(readLines(bridge_ci_meta, warn = FALSE)[1], BRIDGE_CI_VERSION)
    if (bridge_ci_is_current && !FORCE_RERUN) {
      message("    Loading existing bridge CI: ", basename(bridge_ci_csv))
    } else {
      message("    Computing bridge centrality CI from ", N_BOOT, " bootstrap samples...")
      boot_obj <- readRDS(edge_rds)
      bt <- boot_obj$bootTable
      bt_edge <- bt[bt$type == "edge", , drop = FALSE]
      node_names <- colnames(boot_obj$sample$graph)
      n_nodes <- length(node_names)
      str_list <- list()
      ei_list <- list()
      for (boot_name in unique(bt_edge$name)) {
        sub <- bt_edge[bt_edge$name == boot_name, , drop = FALSE]
        mat <- matrix(0, n_nodes, n_nodes, dimnames = list(node_names, node_names))
        for (r in seq_len(nrow(sub))) {
          mat[sub$node1[r], sub$node2[r]] <- sub$value[r]
        }
        mat <- mat + t(mat)
        diag(mat) <- 0
        b <- tryCatch(bridge(mat, communities = communities), error = function(e) NULL)
        if (!is.null(b)) {
          str_list[[as.character(boot_name)]] <- b[["Bridge Strength"]]
          ei_list[[as.character(boot_name)]] <- b[["Bridge Expected Influence (1-step)"]]
        }
      }
      if (length(str_list) > 0 && length(ei_list) > 0) {
        str_mat <- do.call(rbind, str_list)
        ei_mat <- do.call(rbind, ei_list)
        str_ci <- t(apply(str_mat, 2, quantile, probs = c(0.025, 0.975), na.rm = TRUE))
        ei_ci <- t(apply(ei_mat, 2, quantile, probs = c(0.025, 0.975), na.rm = TRUE))
        ci_table <- data.frame(
          Node = colnames(str_mat),
          Bridge_Strength_Mean = colMeans(str_mat, na.rm = TRUE),
          Bridge_Strength_SD = apply(str_mat, 2, sd, na.rm = TRUE),
          CI2.5_Strength = str_ci[, 1],
          CI97.5_Strength = str_ci[, 2],
          Bridge_EI_Mean = colMeans(ei_mat, na.rm = TRUE),
          Bridge_EI_SD = apply(ei_mat, 2, sd, na.rm = TRUE),
          CI2.5_EI = ei_ci[, 1],
          CI97.5_EI = ei_ci[, 2]
        )
        write.csv(ci_table, bridge_ci_csv, row.names = FALSE)
        writeLines(c(
          BRIDGE_CI_VERSION,
          paste("bootstrap_samples_used", length(str_list), sep = "="),
          paste("created", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), sep = "=")
        ), bridge_ci_meta)
        message("    Bridge bootstrap CI saved to ", basename(bridge_ci_csv))
      } else {
        message("    Warning: no valid bridge bootstrap samples")
      }
    }

    pdf(file.path(output_dir, sprintf("cluster%d_edge_weight_CI.pdf", k)), width = 10, height = 8)
    print(plot(boot_edge, labels = FALSE, order = "sample"))
    dev.off()
    pdf(file.path(output_dir, sprintf("cluster%d_edge_difference_test.pdf", k)), width = 10, height = 8)
    print(plot(boot_edge, "edge", plot = "difference", onlyNonZero = TRUE, order = "sample"))
    dev.off()

    case_rds <- file.path(output_dir, sprintf("cluster%d_case_bootstrap.rds", k))
    boot_case <- skip_if_done(case_rds, {
      set.seed(20260424)
      bootnet(net, nBoots = N_BOOT, nCores = N_CORES, type = "case",
              statistics = c("strength", "expectedInfluence",
                             "bridgeStrength", "bridgeExpectedInfluence"),
              communities = communities)
    })
    boot_cases[[as.character(k)]] <- boot_case
    if (!file.exists(case_rds) || FORCE_RERUN) saveRDS(boot_case, case_rds)
    pdf(file.path(output_dir, sprintf("cluster%d_centrality_stability.pdf", k)), width = 8, height = 6)
    print(plot(boot_case, statistics = c("strength", "expectedInfluence")))
    dev.off()
    pdf(file.path(output_dir, sprintf("cluster%d_bridge_centrality_stability.pdf", k)), width = 8, height = 6)
    print(plot(boot_case, statistics = c("bridgeStrength", "bridgeExpectedInfluence")))
    dev.off()
  }

  # ── Network description statistics ────────────────────────
  n_nodes <- ncol(networks[[1]]$graph)
  total_possible <- n_nodes * (n_nodes - 1) / 2
  desc_list <- lapply(names(networks), function(k) {
    gvec <- networks[[k]]$graph[upper.tri(networks[[k]]$graph)]
    nz <- gvec[gvec != 0]
    data.frame(
      Cluster = paste("Cluster", k),
      Nodes = n_nodes,
      Total_edges_possible = total_possible,
      Nonzero_edges = sum(gvec != 0),
      Density = sum(gvec != 0) / total_possible,
      Mean_abs_weight = mean(abs(gvec)),
      Mean_abs_nz_weight = if (length(nz) > 0) mean(abs(nz)) else NA,
      Max_abs_weight = max(abs(gvec)),
      Positive_edges_prop = sum(gvec > 0) / total_possible,
      Negative_edges_prop = sum(gvec < 0) / total_possible
    )
  })
  desc_table <- do.call(rbind, desc_list)
  write.csv(desc_table, file.path(output_dir, "network_description.csv"), row.names = FALSE)

  # ── CS coefficients ────────────────────────────────────────
  cluster_ids <- sort(unique(as.integer(as.character(df$cluster))))
  cs_strength <- sapply(cluster_ids, function(k) corStability(boot_cases[[as.character(k)]], statistics = "strength"))
  cs_ei <- sapply(cluster_ids, function(k) corStability(boot_cases[[as.character(k)]], statistics = "expectedInfluence"))
  cs_bridge_strength <- sapply(cluster_ids, function(k) corStability(boot_cases[[as.character(k)]], statistics = "bridgeStrength"))
  cs_bridge_ei <- sapply(cluster_ids, function(k) corStability(boot_cases[[as.character(k)]], statistics = "bridgeExpectedInfluence"))
  cs_table <- data.frame(
    Cluster = paste("Cluster", cluster_ids),
    N = sapply(cluster_ids, function(k) sum(df$cluster == k)),
    CS_strength = cs_strength,
    CS_expectedInfluence = cs_ei,
    CS_bridgeStrength = cs_bridge_strength,
    CS_bridgeExpectedInfluence = cs_bridge_ei,
    Centrality_stability_ok = cs_strength >= 0.50 & cs_ei >= 0.50,
    Bridge_stability_ok = cs_bridge_strength >= 0.50 & cs_bridge_ei >= 0.50,
    Stability_ok = cs_strength >= 0.50 & cs_ei >= 0.50 &
      cs_bridge_strength >= 0.50 & cs_bridge_ei >= 0.50
  )
  write.csv(cs_table, file.path(output_dir, "cs_coefficients.csv"), row.names = FALSE)
  write.csv(
    cs_table[c("Cluster", "N", "CS_bridgeStrength", "CS_bridgeExpectedInfluence", "Bridge_stability_ok")],
    file.path(output_dir, "bridge_cs_coefficients.csv"),
    row.names = FALSE
  )
  message("CS coefficients (>= 0.50 indicates stable estimation; Epskamp et al., 2018):")
  for (i in seq_len(nrow(cs_table))) {
    message("  ", cs_table$Cluster[i], " — Strength: ", round(cs_strength[i], 3),
            ", EI: ", round(cs_ei[i], 3),
            ", Bridge strength: ", round(cs_bridge_strength[i], 3),
            ", Bridge EI: ", round(cs_bridge_ei[i], 3),
            if (cs_table$Stability_ok[i]) " [OK]" else " [CAUTION: one or more CS coefficients < 0.50]")
  }

  # ── Global strength ────────────────────────────────────────
  s_values <- sapply(networks, function(net) sum(abs(net$graph[upper.tri(net$graph)])))
  write.csv(
    data.frame(Cluster = paste("Cluster", names(s_values)),
               Global_strength = round(as.numeric(s_values), 4)),
    file.path(output_dir, "global_strength.csv"), row.names = FALSE
  )

  # ── Network Comparison Test ────────────────────────────────
  all_pairs <- list(c(1, 2), c(1, 3), c(2, 3))
  nct_summary_file <- file.path(output_dir, "nct_key_statistics.csv")

  if (file.exists(nct_summary_file) && !FORCE_RERUN) {
    message("  NCT results exist — loading ", basename(nct_summary_file))
    nct_tables <- read.csv(nct_summary_file, stringsAsFactors = FALSE)
  } else {
    nct_tables <- data.frame(comparison = character(), metric = character(), value = character())
    for (pair in all_pairs) {
      left <- as.character(pair[1])
      right <- as.character(pair[2])
      nct_result <- NCT(
        networks[[left]], networks[[right]], it = N_BOOT, binary.data = FALSE,
        test.edges = TRUE, test.centrality = TRUE,
        centrality = c("expectedInfluence", "strength"), nodes = "all",
        p.adjust.methods = "BH", verbose = TRUE
      )
      comparison_name <- sprintf("cluster%s_vs_cluster%s", left, right)
      gls_pval <- nct_result[["glstrinv.pval"]]
      nw_pval <- nct_result[["nwinv.pval"]]
      message("  NCT ", comparison_name,
              " — global strength p = ", round(gls_pval, 4),
              ", maximum edge difference p = ", round(nw_pval, 4))

      # Scalar test statistics (global strength, network invariance)
      for (metric in names(nct_result)) {
        value <- nct_result[[metric]]
        if (is.atomic(value) && length(value) == 1) {
          nct_tables <- rbind(nct_tables,
            data.frame(comparison = comparison_name, metric = metric, value = as.character(value)))
        }
      }

      # Edge-wise test results (p-values and test statistics)
      write.csv(nct_result[["einv.pvals"]],
        file.path(output_dir, sprintf("edge_invariance_pval_%s.csv", comparison_name)))
      write.csv(nct_result[["einv.real"]],
        file.path(output_dir, sprintf("edge_invariance_stat_%s.csv", comparison_name)))
      write.csv(nct_result[["diffcen.pval"]],
        file.path(output_dir, sprintf("centrality_invariance_pval_%s.csv", comparison_name)))
      write.csv(nct_result[["diffcen.real"]],
        file.path(output_dir, sprintf("centrality_invariance_stat_%s.csv", comparison_name)))

      # Difference network plot
      diff_mat <- networks[[left]]$graph - networks[[right]]$graph
      diag(diff_mat) <- 0

      # P-value matrix
      pv <- nct_result[["einv.pvals"]]
      nodes <- rownames(diff_mat)

      pval_mat <- matrix(
        1,
        nrow = length(nodes),
        ncol = length(nodes),
        dimnames = list(nodes, nodes)
      )

      if (is.data.frame(pv) && all(c("Var1", "Var2", "p-value") %in% colnames(pv))) {
        for (r in seq_len(nrow(pv))) {
          i <- as.character(pv[["Var1"]][r])
          j <- as.character(pv[["Var2"]][r])
          p <- pv[["p-value"]][r]

          if (i %in% nodes && j %in% nodes) {
            pval_mat[i, j] <- pval_mat[j, i] <- p
          }
        }
      } else {
        pv_mat <- as.matrix(pv)
        common_nodes <- intersect(nodes, rownames(pv_mat))
        common_nodes <- intersect(common_nodes, colnames(pv_mat))
        pval_mat[common_nodes, common_nodes] <- pv_mat[common_nodes, common_nodes]
      }

      sig_mat <- pval_mat < 0.05
      diag(sig_mat) <- FALSE

      sig_diff_plot <- diff_mat
      sig_diff_plot[!sig_mat] <- 0
      diag(sig_diff_plot) <- 0
      max_sig_diff <- max(abs(sig_diff_plot), na.rm = TRUE)
      if (!is.finite(max_sig_diff) || max_sig_diff == 0) max_sig_diff <- 1

      pdf(file.path(output_dir, sprintf("network_difference_%s.pdf", comparison_name)),
          width = 8, height = 6)
      qgraph(
        sig_diff_plot,
        layout = layout,
        posCol = "#E41A1C",
        negCol = "#377EB8",
        title = paste0(comparison_name, " significant edges only"),
        negDashed = TRUE,
        theme = "colorblind",
        minimum = 0,
        cut = 0,
        maximum = max_sig_diff,
        vsize = 6,
        label.cex = 1.2,
        edge.labels = TRUE
      )
      dev.off()
    }  # end for (pair in all_pairs)
    write.csv(nct_tables, file.path(output_dir, "nct_key_statistics.csv"), row.names = FALSE)
  }  # end else (!nct_skip)
  nct_notes <- c(
    "Network Comparison Test interpretation notes.",
    "Global strength invariance and maximum edge-difference network invariance are the primary omnibus tests.",
    "Edge-wise and centrality-wise tests are BH-adjusted exploratory follow-ups and should be interpreted only after considering the omnibus tests, multiple testing, and bootstrap stability.",
    sprintf("Permutation count: %d; smallest attainable p-value: %.6f.", N_BOOT, 1 / (N_BOOT + 1))
  )
  for (comparison_name in unique(nct_tables$comparison)) {
    sub <- nct_tables[nct_tables$comparison == comparison_name, , drop = FALSE]
    gls_p <- suppressWarnings(as.numeric(sub$value[sub$metric == "glstrinv.pval"][1]))
    nw_p <- suppressWarnings(as.numeric(sub$value[sub$metric == "nwinv.pval"][1]))
    if (!is.na(gls_p) && !is.na(nw_p)) {
      nct_notes <- c(nct_notes, sprintf(
        "%s: global strength p = %.6f; maximum edge difference p = %.6f.",
        comparison_name, gls_p, nw_p
      ))
    }
  }
  writeLines(nct_notes, file.path(output_dir, "nct_interpretation_notes.txt"))

  # ── Centrality comparison plot ─────────────────────────────
  cent_list <- list()
  for (k in as.character(sort(as.integer(names(plots))))) {
    cent_auto <- centrality_auto(plots[[k]])
    cent_df <- as.data.frame(cent_auto$node.centrality)[c("ExpectedInfluence", "Strength")]
    cent_df$type <- paste0("Class", k)
    cent_df$name <- rownames(cent_df)
    cent_list[[k]] <- cent_df
  }
  cent_long <- melt(do.call(rbind, cent_list), id.vars = c("name", "type"))

  # ExpectedInfluence
  ei_dat <- filter(cent_long, variable == "ExpectedInfluence") %>%
    mutate(node_order = match(name, {
      filter(cent_long, variable == "ExpectedInfluence") %>%
        group_by(name) %>% summarize(mv = mean(value)) %>% arrange(mv) %>% pull(name)
    }))
  p1 <- ggplot(ei_dat, aes(x = node_order, y = value, color = type, group = type)) +
    geom_line(size = 0.4) + geom_point(size = 1.3, alpha = 0.6) +
    scale_x_continuous(breaks = unique(ei_dat$node_order), labels = unique(ei_dat$name),
                       expand = expansion(add = 0.3)) +
    scale_color_manual(values = CLASS_COLORS) +
    theme_minimal() + coord_flip() +
    theme(legend.position = "none", axis.text.x = element_text(angle = 90, hjust = 1),
          axis.title.x = element_text(size = 9), axis.title.y = element_blank(),
          plot.title = element_text(hjust = 0.5, size = 10)) +
    xlab("Raw expected influence") +
    ggtitle("Expected Influence")

  # Strength
  str_dat <- filter(cent_long, variable == "Strength") %>%
    mutate(node_order = match(name, {
      filter(cent_long, variable == "Strength") %>%
        group_by(name) %>% summarize(mv = mean(value)) %>% arrange(mv) %>% pull(name)
    }))
  p2 <- ggplot(str_dat, aes(x = node_order, y = value, color = type, group = type)) +
    geom_line(size = 0.4) + geom_point(size = 1.3, alpha = 0.6) +
    scale_x_continuous(breaks = unique(str_dat$node_order), labels = unique(str_dat$name),
                       expand = expansion(add = 0.3)) +
    scale_color_manual(values = CLASS_COLORS) +
    theme_minimal() + coord_flip() +
    theme(legend.position = "none", axis.text.x = element_text(angle = 90, hjust = 1),
          axis.title.x = element_text(size = 9), axis.title.y = element_blank(),
          plot.title = element_text(hjust = 0.5, size = 10)) +
    xlab("Raw strength") +
    ggtitle("Strength")

  # Legend
  legend <- cowplot::get_legend(
    ggplot(cent_long, aes(x = name, y = value, color = type, group = type)) +
      geom_line(size = 0.4) + geom_point(size = 1.3, alpha = 0.6) +
      scale_color_manual(values = CLASS_COLORS) +
      theme_minimal() + guides(color = guide_legend(title = NULL)) +
      theme(legend.position = "right")
  )

  pdf(file.path(output_dir, "centrality_comparison.pdf"), width = 10, height = 8)
  grid.arrange(p1, p2, legend, ncol = 3, widths = c(4, 4, 1))
  dev.off()

  # ── Bridge centrality comparison plot ──────────────────────
  bridge_list <- list()
  for (k in as.character(sort(as.integer(names(plots))))) {
    b <- bridges[[k]]
    bridge_list[[k]] <- data.frame(
      BridgeStrength = b[["Bridge Strength"]],
      BridgeEI = b[["Bridge Expected Influence (1-step)"]],
      name = names(b[["Bridge Strength"]]),
      type = paste0("Class", k),
      row.names = NULL
    )
  }
  bridge_long <- melt(do.call(rbind, bridge_list),
                       id.vars = c("name", "type"),
                       variable.name = "bridge_metric", value.name = "value")

  # Bridge Strength
  bs_dat <- filter(bridge_long, bridge_metric == "BridgeStrength") %>%
    mutate(node_order = match(name, {
      filter(bridge_long, bridge_metric == "BridgeStrength") %>%
        group_by(name) %>% summarize(mv = mean(value)) %>% arrange(mv) %>% pull(name)
    }))
  pb1 <- ggplot(bs_dat, aes(x = node_order, y = value, color = type, group = type)) +
    geom_line(size = 0.4) + geom_point(size = 1.3, alpha = 0.6) +
    scale_x_continuous(breaks = unique(bs_dat$node_order), labels = unique(bs_dat$name),
                       expand = expansion(add = 0.3)) +
    scale_color_manual(values = CLASS_COLORS) +
    theme_minimal() + coord_flip() +
    theme(legend.position = "none", axis.text.x = element_text(angle = 90, hjust = 1),
          axis.title.x = element_text(size = 9), axis.title.y = element_blank(),
          plot.title = element_text(hjust = 0.5, size = 10)) +
    xlab("Raw bridge strength") +
    ggtitle("Bridge Strength")

  # Bridge Expected Influence
  bei_dat <- filter(bridge_long, bridge_metric == "BridgeEI") %>%
    mutate(node_order = match(name, {
      filter(bridge_long, bridge_metric == "BridgeEI") %>%
        group_by(name) %>% summarize(mv = mean(value)) %>% arrange(mv) %>% pull(name)
    }))
  pb2 <- ggplot(bei_dat, aes(x = node_order, y = value, color = type, group = type)) +
    geom_line(size = 0.4) + geom_point(size = 1.3, alpha = 0.6) +
    scale_x_continuous(breaks = unique(bei_dat$node_order), labels = unique(bei_dat$name),
                       expand = expansion(add = 0.3)) +
    scale_color_manual(values = CLASS_COLORS) +
    theme_minimal() + coord_flip() +
    theme(legend.position = "none", axis.text.x = element_text(angle = 90, hjust = 1),
          axis.title.x = element_text(size = 9), axis.title.y = element_blank(),
          plot.title = element_text(hjust = 0.5, size = 10)) +
    xlab("Raw bridge expected influence") +
    ggtitle("Bridge Expected Influence (1-step)")

  pdf(file.path(output_dir, "bridge_centrality_comparison.pdf"), width = 12, height = 8)
  grid.arrange(pb1, pb2, legend, ncol = 3, widths = c(4, 4, 1))
  dev.off()

  message("Done: ", analysis_name)
}

# ── Session info ──────────────────────────────────────────
sink("Network_result/session_info.txt")
cat("Session info for network_analysis.R — ",
    format(Sys.time(), "%Y-%m-%d %H:%M"), "\n\n", sep = "")
cat("Analysis configuration\n")
cat("BEST_CLASS =", BEST_CLASS, "\n")
cat("N_BOOT =", N_BOOT, "\n")
cat("N_CORES =", N_CORES, "\n")
cat("FORCE_RERUN =", FORCE_RERUN, "\n")
cat("BRIDGE_CI_VERSION =", BRIDGE_CI_VERSION, "\n\n")
print(sessionInfo())
sink()
