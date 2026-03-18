import marimo

__generated_with = "0.21.0"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    from datetime import datetime, timezone

    import polars as pl
    import polars.selectors as cs
    from great_tables import GT, loc, style
    from natsort import natsort_keygen

    return GT, Path, cs, datetime, loc, pl, style, timezone


@app.cell
def _(Path):
    CSV_PATH = Path("polars_compare.csv")
    return (CSV_PATH,)


@app.cell
def _(CSV_PATH, Path, pl):
    def load_comparison_lf(file_path: Path | None=None) -> pl.LazyFrame:
        csvpath = file_path if file_path is not None else CSV_PATH
        id_schema = {'id': pl.String, 'timestamp': pl.Datetime(time_zone='UTC')}
        if not csvpath.is_file():
            lf = pl.LazyFrame({'id': [], 'timestamp': []}, id_schema)
        else:
            lf = pl.scan_csv(csvpath, schema_overrides=id_schema, null_values='')
        _schema = lf.collect_schema()
        match_col_names = [col_name for col_name in _schema if '.is_match' in col_name]
        match_cast_exprs = [pl.col(col_name).cast(pl.Boolean) for col_name in match_col_names]
        lf = lf.with_columns(match_cast_exprs)
        return lf

    return (load_comparison_lf,)


@app.cell
def _(load_comparison_lf):
    lf = load_comparison_lf()
    return (lf,)


@app.cell
def _(lf):
    lf.collect_schema()
    return


@app.cell
def _(lf):
    lf.collect()
    return


@app.cell
def _(cs, datetime, lf, pl, timezone):
    HOURS_IN_DAY = 24
    MINS_IN_HOUR = 60
    MINS_IN_DAY = MINS_IN_HOUR * HOURS_IN_DAY

    def _cast(val: pl.Expr) -> pl.Expr:
        return val.cast(pl.Int64).cast(pl.Utf8)
    now = pl.lit(datetime.now(timezone.utc), dtype=pl.Datetime)
    mins_since_last = pl.col('last_run_age').dt.total_minutes()
    _schema = lf.collect_schema()
    struct_col_names = [col_name for (col_name, dtype) in _schema.items() if isinstance(dtype, pl.Struct)]
    match_lf = lf.with_columns(pl.all_horizontal(cs.ends_with('.is_match') & cs.boolean()).fill_null(True).alias('all_match')).with_columns(pl.col('timestamp').dt.convert_time_zone('America/Vancouver').dt.to_string('%Y-%m-%d %H:%M:%S.%3f').alias('last_run')).with_columns((now - pl.col('timestamp')).alias('last_run_age')).with_columns(pl.when(mins_since_last >= MINS_IN_DAY).then(_cast(mins_since_last // MINS_IN_DAY) + 'd ' + _cast(mins_since_last % MINS_IN_DAY // MINS_IN_HOUR) + 'h').when(mins_since_last >= MINS_IN_HOUR).then(_cast(mins_since_last // MINS_IN_HOUR) + 'h ' + _cast(mins_since_last % MINS_IN_HOUR) + 'm').otherwise(_cast(mins_since_last) + 'm').alias('last_run_age_str')).sort('timestamp')
    return (match_lf,)


@app.cell
def _(match_lf):
    match_lf.collect()
    return


@app.cell
def _(GT, loc, pl, style):
    def format_run_match_table(gtable: GT, header_text: str):
        gtable = (
            gtable.tab_header(header_text)
            .tab_options(
                # table_width="50%",
                table_body_vlines_style="solid",
                table_body_vlines_color="#D3D3D3",
                column_labels_padding_horizontal="25px",
                data_row_padding_horizontal="20px",
            )
            .tab_spanner(
                label="Last run",
                columns=["last_run", "last_run_age_str"]
            )
            .cols_label(
                cases={
                    "id": "Exp. ID",
                    "last_run": "Date/time",
                    "last_run_age_str": "Age",
                    "all_match": "Legacy match",
                }
            )
            .cols_hide(
                columns="timestamp"
            )
            .fmt_datetime(
                columns="last_run", date_style="m_day_year", time_style="h_m_p"
            )
            .tab_style(
                style=[style.fill(color="red"), style.text(color="white")],
                locations=loc.body(rows=pl.col("all_match").not_()),
            )
            .tab_style(
                style=[
                    style.fill(color="green"),
                    style.text(color="white", weight="bold"),
                ],
                locations=loc.body(rows=pl.col("all_match")),
            )
            # .fmt(
            #     columns="all_match",
            #     fns=lambda x: "✅" if x else "❌",
            #     is_substitution=True,
            # )
            .fmt_tf(
                columns="all_match",
                tf_style="check-mark"
            )
        )
        return gtable

    return (format_run_match_table,)


@app.cell
def _(GT, format_run_match_table, match_lf, pl):
    last_by_id_lf = (
        match_lf.group_by("id")
        .last()
        .sort("id")
        .select(["id", "last_run", "timestamp", "last_run_age_str", "all_match"])
    )
    # print(test_lf.explain())
    last_by_id_df = last_by_id_lf.collect()
    last_by_id_gt = last_by_id_df.style.pipe(
        format_run_match_table, "Last run (by ID)"
    )

    match_series = last_by_id_df.get_column("all_match")
    success_count = match_series.sum()
    failure_count = match_series.not_().sum()
    total_count = last_by_id_df.get_column("id").count()
    success_percent = success_count / total_count
    failure_percent = failure_count / total_count

    latest_run_df = last_by_id_df.filter(pl.col("timestamp") == pl.col("timestamp").max())
    earliest_run_df = last_by_id_df.filter(pl.col("timestamp") == pl.col("timestamp").min())
    latest_run_exp = latest_run_df.get_column("id").item(0)
    earliest_run_exp = earliest_run_df.get_column("id").item(0)
    latest_run_dt = latest_run_df.get_column("last_run").item(0)
    earliest_run_dt = earliest_run_df.get_column("last_run").item(0)
    latest_run_age = latest_run_df.get_column("last_run_age_str").item(0)
    earliest_run_age = earliest_run_df.get_column("last_run_age_str").item(0)

    success_summary = pl.DataFrame(
        {
            "variable": ["Success (%)", "Failure (%)"],
            "value": [success_percent, failure_percent],
        }
    ).with_columns(pl.col("value").alias("plot"))
    success_summary_gt = (
        GT(success_summary)
        .tab_header("Success summary")
        .fmt_percent(columns="value", decimals=1)
        .fmt_nanoplot(columns="plot", plot_type="bar", reference_line=1)
        .tab_options(column_labels_hidden=True)
    )

    run_date_summary = pl.DataFrame(
        {
            "variable": ["Earliest run", "Latest run"],
            "id": [earliest_run_exp, latest_run_exp],
            "last_run": [earliest_run_dt, latest_run_dt],
            "last_run_age": [earliest_run_age, latest_run_age]
        }
    )
    run_date_summary_gt = (
        GT(run_date_summary)
        .tab_header("Run age summary")
        .fmt_datetime(columns="last_run", date_style="m_day_year", time_style="h_m_p")
        .tab_spanner(
            label="Last run",
            columns=["last_run", "last_run_age"]
        )
        .cols_label(
            cases={
                "id": "Exp. ID",
                "last_run": "Date/time",
                "last_run_age": "Age"
                # "all_match": "Legacy match",
            }
        )
        .cols_hide(columns=["variable"])
    )

    success_summary_gt.show()
    run_date_summary_gt.show()
    last_by_id_gt.show()
    return (last_by_id_lf,)


@app.cell
def _(format_run_match_table, last_by_id_lf, pl):
    bad_results_lf = last_by_id_lf.filter(~pl.col("all_match"))
    bad_results_gt = bad_results_lf.collect().style.pipe(format_run_match_table, "Bad results (by ID)")
    bad_results_gt
    return


@app.cell
def _(format_run_match_table, match_lf):
    last_n = 5
    last_n_runs = match_lf.tail(last_n).select(["id", "last_run", "last_run_age_str", "timestamp", "all_match"])
    last_runs_styled = (
        last_n_runs.collect()
        .style.pipe(format_run_match_table, f"Last {last_n} runs")
        .cols_move_to_start(columns=["last_run", "last_run_age_str"])
    )
    # last_runs_df = last_n_runs.collect()
    # last_runs_styled = (
    #     last_runs_df.style.tab_header(f"Last {last_n} runs")
    #     .tab_options(
    #         # table_width="50%",
    #         table_body_vlines_style="solid",
    #         table_body_vlines_color="#D3D3D3",
    #         column_labels_padding_horizontal="25px",
    #         data_row_padding_horizontal="20px",
    #     )
    #     .cols_label(
    #         cases={
    #             "id": "Exp. ID",
    #             "last_run": "Last run on",
    #             "all_match": "Legacy match",
    #         }
    #     )
    #     .cols_move_to_start(columns="last_run")
    #     .fmt_datetime(
    #         columns="last_run", date_style="m_day_year", time_style="h_m_p"
    #     )
    #     .tab_style(
    #         style=[style.fill(color="red"), style.text(color="white")],
    #         locations=loc.body(rows=pl.col("all_match").not_()),
    #     )
    #     .tab_style(
    #         style=[
    #             style.fill(color="green"),
    #             style.text(color="white", weight="bold"),
    #         ],
    #         locations=loc.body(rows=pl.col("all_match")),
    #     )
    #     .fmt(
    #         columns="all_match",
    #         fns=lambda x: "✅" if x else "❌",
    #         is_substitution=True,
    #     )
    # )
    last_runs_styled
    return


@app.cell
def _(cs, lf, loc, pl, style):
    def map_step_name(name: str) -> str:
        (step_name, _) = name.split('.')
        return step_name

    def map_rle_name(name: str) -> str:
        step_name = map_step_name(name)
        return f'{step_name}.rle'

    def map_streak_name(name: str) -> str:
        step_name = map_step_name(name)
        return f'{step_name}.streak'

    def make_display_col_name(name: str) -> str:
        step_name = map_step_name(name)
        (step_number, step_text) = step_name.split('-')
        return f'Step {step_number}: {step_text.capitalize()}'
    win_streak_lf = lf.with_columns((cs.ends_with('.is_match') & cs.boolean()).rle().last().name.map(map_rle_name)).select(cs.ends_with('.rle') & cs.struct()).last()
    _schema = win_streak_lf.collect_schema()
    unnest_cols = [col_name for (col_name, dtype) in _schema.items() if isinstance(dtype, pl.Struct)]
    win_streak_lf = win_streak_lf.with_columns([pl.when(pl.col(c).struct.field('value')).then(pl.col(c).struct.field('len')).otherwise(0).alias(c.replace('.rle', '.streak')) for c in unnest_cols]).select(cs.ends_with('.streak'))
    col_rename_dict = {map_streak_name(col_name): make_display_col_name(col_name) for col_name in unnest_cols}
    win_streak_df = win_streak_lf.collect().unpivot(variable_name='Step', value_name='Streak').with_columns(pl.col('Step').map_elements(make_display_col_name, return_dtype=pl.String))
    win_streak_styled = win_streak_df.style.tab_style(style=[style.fill(color='green'), style.text(color='white')], locations=loc.body(rows=pl.col('Streak').ge(50))).tab_options(table_body_vlines_style='solid', table_body_vlines_color='#D3D3D3', column_labels_padding_horizontal='25px', data_row_padding_horizontal='20px').tab_header('Step success streak')
    # schema
    # print(unnest_cols)
    # print(col_rename_dict)
    # win_streak_df
    win_streak_styled  # .tab_style(  #     style=style.borders(sides="right", color="#D3D3D3"), locations=loc.body(columns="Step")  # )  # .tab_style(  #     style=style.borders(sides="right", color="#D3D3D3"),  #     locations=loc.column_labels(columns="Step"),  # )  # table_width="50%",
    return


@app.cell
def _():
    # TODO find datasets that haven't been tested on all existing steps
    return


if __name__ == "__main__":
    app.run()
