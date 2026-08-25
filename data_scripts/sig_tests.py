import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
import scikit_posthocs as sp


path = "C:/Users/Amis Thysia/Documents/Napier Masters Stuff/Dissertation/sig_test_new_script_test/"
file = "testdata"
df = pd.read_csv(path + file + ".csv")

# list of metrics/columns in csv

metrics = [
    "coverage", "coverage1700", "max", "max1700", "mean", "mean1700", "qd", "qd1700", "broken", "broken1700", "brokenpersim"
]
"""
metrics = [
    "coverage", "max", "mean", "qd", "broken"
]
"""
"""
metrics = [
    "coveragepersim", "maxpersim", "meanpersim", "qdpersim"
]
"""

parametric_csv_rows = []
mixed_csv_rows = []

# activate to force nonparametric route (MWU/KW) all the way through, effectively same as mixed but assuming all non-normal
# if set to true, mixed_results will actually just be all nonparametric
force_nonparametric = True

# what to group data by, i.e. what's the things staying the same when we make comparisons
# so if comparing algorithms, this would be morphology, sensor_mode
# if comparing sensor modes, this would be algorithm, morphology
groupby1_str, groupby2_str = "morphology", "sensor_mode"
config_groups = df.groupby([groupby1_str, groupby2_str])

# and the third thing, the one being compared, goes here
compgroup_str = "algorithm"

for (group1, group2), config_df in config_groups:
    for metric in metrics:

        config_df = config_df.copy()

        groups_raw = {}
        means = {}
        medians = {}
        for compgroup in config_df[compgroup_str].unique():
            valid_raw = (
                config_df[config_df[compgroup_str] == compgroup][metric]
                .dropna()
                .values
            )
            if len(valid_raw) > 0:
                groups_raw[compgroup] = valid_raw
                means[compgroup] = np.mean(valid_raw)
                medians[compgroup] = np.median(valid_raw)

        num_groups = len(groups_raw)

        # if < 2 groups skip, no statistical comparison to make
        if num_groups < 2:
            continue
        
        # normality testing for mixed route
        all_groups_normal = True
        for compgroup, data in groups_raw.items():
            if len(data) >= 3:
                _, shapiro_p = stats.shapiro(data)
                if shapiro_p < 0.05:
                    all_groups_normal = False
            else:
                all_groups_normal = False
        if all_groups_normal and not force_nonparametric:
            mixed_route_str = "Parametric Mixed"
        else: mixed_route_str = "Non-Parametric Mixed"

        # IF TWO TO COMPARE:
        if num_groups == 2:
            compgroup1, compgroup2 = list(groups_raw.keys())

            # Parametric: Welch's t-test
            _, p_val_parametric = stats.ttest_ind(
                groups_raw[compgroup1], groups_raw[compgroup2], equal_var=False
            )
            is_sig_parametric = p_val_parametric < 0.05

            parametric_winner = compgroup1 if means[compgroup1] > means[compgroup2] else compgroup2

            parametric_csv_rows.append(
                {
                    groupby1_str: group1,
                    groupby2_str: group2,
                    "metric": metric,
                    "test_pipeline": "Parametric Welch",
                    "comparison_type": "Welch t-test",
                    "group_a": compgroup1,
                    "group_b": compgroup2,
                    "p_value": round(p_val_parametric, 5),
                    "is_significant": is_sig_parametric,
                    "higher_performing_group": parametric_winner if is_sig_parametric else "None",
                }
            )

            # Mixed: t-test/mann-whitney
            if all_groups_normal and not force_nonparametric:
                _, p_val_mixed = stats.ttest_ind(groups_raw[compgroup1], groups_raw[compgroup2])
                comp_type = "t-test"
                mixed_winner = compgroup1 if means[compgroup1] > means[compgroup2] else compgroup2
            else:
                # for nonparametric, we need to look at medians not means for winner as they compare
                # means of *ranks* (which is basically the median)
                _, p_val_mixed = stats.mannwhitneyu(groups_raw[compgroup1], groups_raw[compgroup2])
                comp_type = "Mann-Whitney U"
                mixed_winner = compgroup1 if medians[compgroup1] > medians[compgroup2] else compgroup2

            is_sig_mixed = p_val_mixed < 0.05

            mixed_csv_rows.append(
                {
                    groupby1_str: group1,
                    groupby2_str: group2,
                    "metric": metric,
                    "test_route": mixed_route_str,
                    "comparison_type": comp_type,
                    "group_a": compgroup1,
                    "group_b": compgroup2,
                    "p_value": round(p_val_mixed, 5),
                    "is_significant": is_sig_mixed,
                    "higher_performing_group": mixed_winner if is_sig_mixed else "None",
                }
            )


        # IF >2 TO COMPARE:
        else:
            clean_df = config_df.dropna(subset=[metric])

            # Parametric:

            # Welch's ANOVA
            welch_result = pg.welch_anova(
                data=clean_df, dv=metric, between=compgroup_str
            )
            p_anova = welch_result["p_unc"].values[0]
            is_welch_anova_sig = p_anova < 0.05

            parametric_csv_rows.append(
                {
                    groupby1_str: group1,
                    groupby2_str: group2,
                    "metric": metric,
                    "test_route": "Parametric Welch",
                    "comparison_type": "Welch ANOVA",
                    "group_a": "ALL",
                    "group_b": "ALL",
                    "p_value": round(p_anova, 5),
                    "is_significant": is_welch_anova_sig,
                    "higher_performing_group": "N/A",
                }
            )

            # If Welch ANOVA overall significant, do games howell pairwise comparisons
            if is_welch_anova_sig:
                gh_result = pg.pairwise_gameshowell(
                    data=clean_df, dv=metric, between="algorithm"
                )

                for _, row in gh_result.iterrows():
                    a1, a2 = row["A"], row["B"]

                    if "p-tukey" in row:
                        p_pair = row["p-tukey"] 
                    else:
                        p_pair = row["pval"]
                    is_pair_sig = p_pair < 0.05

                    winner = a1 if means[a1] > means[a2] else a2

                    parametric_csv_rows.append(
                        {
                            groupby1_str: group1,
                            groupby2_str: group2,
                            "metric": metric,
                            "test_route": "Parametric Welch",
                            "comparison_type": "Pairwise Games-Howell",
                            "group_a": a1,
                            "group_b": a2,
                            "p_value": round(p_pair, 5),
                            "is_significant": is_pair_sig,
                            "higher_performing_group": (
                                winner if is_pair_sig else "None"
                            ),
                        }
                    )



            # MIXED:
            if all_groups_normal and not force_nonparametric:
                _, p_mixedmulti = stats.f_oneway(*groups_raw.values())
                mixedmulti_type = "ANOVA"
            else:
                _, p_mixedmulti = stats.kruskal(*groups_raw.values())
                mixedmulti_type = "Kruskal-Wallis"

            is_mixedmulti_sig = p_mixedmulti < 0.05
            mixed_csv_rows.append(
                {
                    groupby1_str: group1,
                    groupby2_str: group2,
                    "metric": metric,
                    "test_route": mixed_route_str,
                    "comparison_type": mixedmulti_type,
                    "group_a": "ALL",
                    "group_b": "ALL",
                    "p_value": round(p_mixedmulti, 5),
                    "is_significant": is_mixedmulti_sig,
                    "higher_performing_group": "N/A",
                }
            )

            # pairwise comparisons if significant multiway; Tukey HSD for ANOVA/normal, Dunns for KW
            if is_mixedmulti_sig:
                compgroup_list = list(groups_raw.keys())

                if all_groups_normal and not force_nonparametric:
                    # Tukey HSD
                    tukey_res = stats.tukey_hsd(*groups_raw.values())
                    for i in range(len(compgroup_list)):
                        for j in range(i + 1, len(compgroup_list)):
                            a1, a2 = compgroup_list[i], compgroup_list[j]
                            p_pair = tukey_res.pvalue[i, j]
                            is_pair_sig = p_pair < 0.05
                            winner = a1 if means[a1] > means[a2] else a2

                            mixed_csv_rows.append(
                                {
                                    groupby1_str: group1,
                                    groupby2_str: group2,
                                    "metric": metric,
                                    "test_route": mixed_route_str,
                                    "comparison_type": "Pairwise Tukey HSD",
                                    "group_a": a1,
                                    "group_b": a2,
                                    "p_value": round(p_pair, 5),
                                    "is_significant": is_pair_sig,
                                    "higher_performing_group": (
                                        winner if is_pair_sig else "None"
                                    ),
                                }
                            )
                else:
                    # Dunn's
                    clean_config_df = config_df.dropna(subset=[metric])
                    p_matrix = sp.posthoc_dunn(
                        clean_config_df,
                        val_col=metric,
                        group_col=compgroup_str,
                        p_adjust="holm",
                    )
                    for i in range(len(compgroup_list)):
                        for j in range(i + 1, len(compgroup_list)):
                            a1, a2 = compgroup_list[i], compgroup_list[j]
                            p_pair = p_matrix.loc[a1, a2]
                            is_pair_sig = p_pair < 0.05
                            winner = a1 if medians[a1] > medians[a2] else a2

                            mixed_csv_rows.append(
                                {
                                    groupby1_str: group1,
                                    groupby2_str: group2,
                                    "metric": metric,
                                    "test_route": mixed_route_str,
                                    "comparison_type": "Pairwise Dunn",
                                    "group_a": a1,
                                    "group_b": a2,
                                    "p_value": round(p_pair, 5),
                                    "is_significant": is_pair_sig,
                                    "higher_performing_group": (
                                        winner if is_pair_sig else "None"
                                    ),
                                }
                            )



parametric_results_df = pd.DataFrame(parametric_csv_rows)
parametric_results_df.to_csv(path+file+"_parametric_results.csv", index=False)

# note: if force_nonparametric was set to True at top, mixed results will actually be the nonparametric results
# if want all three just run twice with the option enabled and disabled 
# (ensuring not to overwrite the mixed_results CSV from the first run)
mixed_results_df = pd.DataFrame(mixed_csv_rows)
mixed_results_df.to_csv(path+file+"_mixed_results.csv", index=False)
