"""
Created on Tue Mar 10 16:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for evaluating the split correction pipeline

"""


# --- Precision-Recall Curves ---
def compute_precision_recall(gt_graphs, fragment_graphs, labels, proposal_type):
    # Load data
    csv_path = os.path.join(s3_prefix, "proposal_summary.csv")
    proposals_df = data_util.load_proposals_df(csv_path, proposal_type)
    results_df = create_results_df()

    # Compute metrics
    compute_splits_and_merges(
        gt_graphs, fragment_graphs, labels, proposals_df, results_df
    )
    compute_precision_recall_from_df(results_df)

    # Save results
    suffix = ("_" + proposal_type) if isinstance(proposal_type, str) else ""
    path = os.path.join(output_dir, f"metrics_varying_threshold{suffix}.csv")
    results_df.to_csv(path)



def compute_precision_recall_from_df(results_df):
    initial_merges = results_df.loc[1.0, "# Merges"]
    initial_splits = results_df.loc[1.0, "# Splits"]
    for t in results_df.index:
        tp = initial_splits - results_df.loc[t, "# Splits"]
        fp = results_df.loc[t, "# Merges"] - initial_merges

        precision = 1 - fp / (fp + tp + 1e-5)
        recall = tp / (initial_splits + 1e-5)
        f1 = 2 * precision * recall / (precision + recall)

        results_df.loc[t, "Precision"] = round(precision, 4)
        results_df.loc[t, "Recall"] = round(recall, 4)
        results_df.loc[t, "F1"] = round(f1, 4)

    print(results_df)


def compute_splits_and_merges(
    gt_graphs, fragment_graphs, labels, proposals_df, results_df
):
    merge_count_metric = MergeCountMetric(verbose=False)
    split_count_metric = SplitCountMetric(verbose=False)
    for t in tqdm(results_df.index, desc="Precision-Recall Curves"):
        # Build graphs
        gt_graphs_t, fragment_graphs_t = data_util.build_graphs_at_threshold(
            t,
            deepcopy(gt_graphs),
            deepcopy(fragment_graphs),
            labels,
            proposals_df
        )

        # Count merges (if applicable)
        if t > 0.2:
            merge_cnts = merge_count_metric(gt_graphs_t, fragment_graphs_t)
            results_df.loc[t, "# Merges"] = merge_cnts["# Merges"].sum()

        # Count splits
        split_cnts = split_count_metric(gt_graphs_t)
        results_df.loc[t, "# Splits"] = split_cnts["# Splits"].sum()
