"""report merge with sensitivity."""

import pandas as pd


def get_report(bucket: str, sensitivity_parquet: str) -> pd.DataFrame:
    """Temporary function to merge elasticity and sensitivity report."""
    results = []

    df_s = pd.read_parquet(sensitivity_parquet)
    df_report = pd.read_csv(
        f"s3://{bucket}/data_science/eval_results/elasticity/elasticity_report_2024-03-01.csv"
    )
    sensitivity_client = (
        pd.concat([df_report[["client_key", "channel"]], df_s[["client_key", "channel"]]])
        .groupby(["client_key", "channel"])
        .size()
        .reset_index()[["client_key", "channel"]]
    )

    for client_key, channel in zip(
        sensitivity_client["client_key"].tolist(),
        sensitivity_client["channel"].tolist(),
    ):
        df_s_client = df_s[(df_s["client_key"] == client_key) & (df_s["channel"] == channel)]
        uid_sensitivity = df_s_client["uid"].tolist()
        try:
            df_e_client = pd.read_csv(
                f"s3://{bucket}/data_science/eval_results/elasticity/elasticity_{client_key}_{channel}_2024-03-01.csv"
            )
            uid_elasticity = df_e_client[df_e_client["quality_test"]]["uid"].tolist()
        except Exception:
            uid_elasticity = []

        nb_both = len(list(set(uid_sensitivity).intersection(uid_elasticity)))
        nb_at_least_one = len(set(uid_sensitivity + uid_elasticity))
        nb_sensitivity = len(uid_sensitivity)
        # nb_elasticity = len(uid_elasticity)

        results.append(
            {
                "client_key": client_key,
                "channel": channel,
                "uid_with_both": nb_both,
                "uid_with_at_least_one": nb_at_least_one,
                "uid_with_sensitivity": nb_sensitivity,
            }
        )

    result_df = pd.DataFrame(results)

    df_report = pd.read_csv(
        f"s3://{bucket}/data_science/eval_results/elasticity/elasticity_report_2024-03-01.csv"
    )
    del df_report["error"]
    df_report_merge = df_report.merge(result_df, on=["client_key", "channel"], how="outer")
    df_report_merge["sensitivity_from_total"] = round(
        df_report_merge["uid_with_sensitivity"] * 100 / df_report_merge["total_uid"], 2
    )
    df_report_merge["both_from_total"] = round(
        df_report_merge["uid_with_both"] * 100 / df_report_merge["total_uid"], 2
    )
    df_report_merge["at_least_one_from_total"] = round(
        df_report_merge["uid_with_at_least_one"] * 100 / df_report_merge["total_uid"], 2
    )
    df_report_merge = df_report_merge.rename(
        columns={
            "uids_from_total": "elasticity_from_total",
            "uids_from_total_with_data": "elasticity_from_total_with_data",
        }
    )

    df_inventory = df_report_merge[
        [
            "client_key",
            "channel",
            "total_uid",
            "uid_with_both",
            "uid_with_at_least_one",
            "uid_with_elasticity",
            "uid_with_sensitivity",
            "both_from_total",
            "at_least_one_from_total",
            "elasticity_from_total",
            "elasticity_from_total_with_data",
            "sensitivity_from_total",
        ]
    ]
    df_inventory.to_csv(f"{bucket}_inventory_elasticity_sensitivity_March2024.csv", index=False)
    return df_inventory


# import pandas as pd
# from pyathena import connect
# from pyathena.pandas.cursor import PandasCursor
# from time import time
#
# us
# region = "us-east-1"
# output_dir = "s3://noam-test-glue-use1/athena/output/"
# eu
# region = "eu-central-1"
# output_dir = "s3://noam-test-glue-euc1/athena/output/"
#
# start = time()
# df_list = []
# for client_key in client_keys_list:
#     print(f"Now processing {client_key}")
#     t0 = time()

#     query = f"""SELECT uid,
#             channel,
#             MAX(element.name) AS sensitivity_attr_name,
#             MAX(element.value) AS sensitivity_attr_value
#         FROM AwsDataCatalog.analytics.client_key_{client_key}, UNNEST(attrs) AS t(element)
#         WHERE element.name = 'qlia_competition_sensitive'
#             AND (element.value IS NOT NULL OR CAST(element.value AS VARCHAR) != '')
#             AND date > '2024-01-01'
#         GROUP BY uid, channel;"""

#     cursor = connect(
#         s3_staging_dir=output_dir, region_name=region, cursor_class=PandasCursor
#     ).cursor()
#     df = cursor.execute(query).as_pandas()
#     if not df.empty:
#         df.insert(1, "client_key", [client_key] * df.shape[0])
#         print(df.info())
#         print(f"Number of channels = {df['channel'].nunique()}")
#         df_list.append(df)
#     else:
#         print("No data found")
#     print(f"Finished processing {client_key}; Duration = {(time() - t0):.3f} sec\n")
# print(f"Total duration = {(time() - start):.3f} sec")
