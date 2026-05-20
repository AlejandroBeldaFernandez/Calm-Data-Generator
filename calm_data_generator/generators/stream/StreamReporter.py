"""
Static, File-Based Synthetic Data Reporter

This module provides the StreamReporter class, designed to generate a detailed, static report
for a single synthetic dataset using YData Profiling and Plotly visualizations.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from calm_data_generator.generators.configs import ReportConfig
from calm_data_generator.reports.base import BaseReporter
from calm_data_generator.reports.ExternalReporter import ExternalReporter
from calm_data_generator.reports.LocalIndexGenerator import LocalIndexGenerator
from calm_data_generator.reports.Visualizer import Visualizer

logger = logging.getLogger("StreamReporter")


class StreamReporter(BaseReporter):
    """
    Generates a static, file-based report analyzing the properties of a synthetic dataset.
    Uses YData Profiling for detailed analysis and Plotly for interactive visualizations.
    """

    def __init__(self, verbose: bool = True, minimal_report: bool = False):
        """
        Initializes the StreamReporter.

        Args:
            verbose (bool): If True, prints progress messages to the console.
            minimal (bool): If True, generates minimal reports (faster, no correlations).
        """
        super().__init__(verbose=verbose, minimal=minimal_report)

    def generate_report(
        self,
        synthetic_df: pd.DataFrame,
        generator_name: str,
        output_dir: str,
        target_column: Optional[str] = None,
        block_column: Optional[str] = None,
        focus_cols: Optional[List[str]] = None,
        drift_config: Optional[Dict[str, Any]] = None,
        time_col: Optional[str] = None,
        per_block_external_reports: bool = False,
        resample_rule: Optional[Union[str, int]] = None,
        constraints_stats: Optional[Dict[str, int]] = None,
        sequence_config: Optional[Dict] = None,
        report_config: Optional[Union[ReportConfig, Dict]] = None,
    ) -> None:
        """
        Generates a comprehensive file-based report for the synthetic dataset.
        Can use ReportConfig or individual arguments.
        """
        # Resolve Configuration
        if report_config:
            if isinstance(report_config, dict):
                report_config = ReportConfig(**report_config)
        else:
            report_config = ReportConfig(
                output_dir=output_dir,
                target_column=target_column,
                block_column=block_column,
                focus_columns=focus_cols,
                time_col=time_col,
                resample_rule=resample_rule,
                constraints_stats=constraints_stats,
                sequence_config=sequence_config,
                per_block_external_reports=per_block_external_reports,
                auto_report=True,
                # minimal? StreamReporter init has minimal_report.
                minimal=self.minimal,
            )

        # Override config with explicit non-None args (simple merge)
        if output_dir != report_config.output_dir and output_dir != "output":
            report_config.output_dir = output_dir

        # Use config values
        output_dir = report_config.output_dir
        target_column = report_config.target_column
        block_column = report_config.block_column
        focus_cols = report_config.focus_columns
        time_col = report_config.time_col
        resample_rule = report_config.resample_rule
        per_block_external_reports = report_config.per_block_external_reports

        if self.verbose:
            logger.info(
                "SYNTHETIC DATA REPORT | Generator: %s | Timestamp: %s",
                generator_name,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

        os.makedirs(output_dir, exist_ok=True)

        # Determine time column
        final_time_col = (
            time_col
            if time_col and time_col in synthetic_df.columns
            else "timestamp"
            if "timestamp" in synthetic_df.columns
            else None
        )

        # === Resampling / Aggregation Logic ===
        df_for_report = synthetic_df.copy()

        if resample_rule is not None:
            df_for_report = self._apply_resampling(
                df_for_report, final_time_col, block_column, resample_rule
            )

        # === Generate YData Profile ===
        if self.verbose:
            logger.info("Generating YData Profile Report...")

        ExternalReporter.generate_profile(
            df=df_for_report,
            output_dir=output_dir,
            filename="generated_profile.html",
            time_col=final_time_col,
            block_col=block_column,
            title=f"{generator_name} - Data Profile",
        )

        # === Generate Plotly Visualizations ===
        if self.verbose:
            logger.info("Generating Plotly Visualizations...")

        # Density Plots
        Visualizer.generate_density_plots(
            df=df_for_report,
            output_dir=output_dir,
            columns=focus_cols,
            color_col=target_column,
        )

        # Combined PCA + UMAP
        Visualizer.generate_dimensionality_plot(
            df=df_for_report,
            output_dir=output_dir,
            color_col=target_column,
        )

        # === Per-Block Analysis ===
        if (
            block_column
            and block_column in synthetic_df.columns
            and per_block_external_reports
        ):
            if self.verbose:
                logger.info("Generating per-block reports...")

            unique_blocks = sorted(synthetic_df[block_column].unique(), key=str)

            for block_id in unique_blocks:
                block_output_dir = os.path.join(output_dir, f"block_{block_id}_plots")
                os.makedirs(block_output_dir, exist_ok=True)

                block_df = synthetic_df[synthetic_df[block_column] == block_id]

                # YData Profile for block
                ExternalReporter.generate_profile(
                    df=block_df.reset_index(drop=True),
                    output_dir=block_output_dir,
                    title=f"Block {block_id} Profile",
                    time_col=time_col,
                )

                # Plotly plots for block
                Visualizer.generate_density_plots(
                    df=block_df,
                    output_dir=block_output_dir,
                    columns=focus_cols,
                    color_col=target_column,
                )

                # Dashboard for block
                LocalIndexGenerator.create_index(block_output_dir)

        # === Generate Dashboard ===
        LocalIndexGenerator.create_index(output_dir)

        if self.verbose:
            logger.info("Report generated at: %s", output_dir)

    def _apply_resampling(
        self,
        df: pd.DataFrame,
        time_col: Optional[str],
        block_column: Optional[str],
        resample_rule: Union[str, int],
    ) -> pd.DataFrame:
        """
        Applies resampling/aggregation based on time or block column.
        """
        if self.verbose:
            logger.info("Applying resampling rule: %s", resample_rule)

        def agg_mode(x):
            m = x.mode()
            return m.iloc[0] if not m.empty else np.nan

        exclude_cols = [c for c in [time_col, block_column] if c]
        cols_to_agg = [c for c in df.columns if c not in exclude_cols]

        agg_dict = {}
        for col in cols_to_agg:
            if pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = "mean"
            else:
                agg_dict[col] = agg_mode

        if time_col and time_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

            df = df.set_index(time_col)
            df = df.resample(resample_rule).agg(agg_dict).reset_index()
            df = df.dropna(how="all")

        elif (
            block_column
            and block_column in df.columns
            and isinstance(resample_rule, int)
        ):
            df["_block_group"] = df[block_column] // resample_rule
            df = df.groupby("_block_group").agg(agg_dict).reset_index(drop=True)

        return df
