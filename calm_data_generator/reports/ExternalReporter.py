"""
External Reporter Module
Wraps YData Profiling to generate advanced HTML reports.
"""

import logging
import os
from typing import Optional

import pandas as pd

# Initializing logger
logger = logging.getLogger("ExternalReporter")

try:
    from ydata_profiling import ProfileReport

    YDATA_AVAILABLE = True
except ImportError:
    YDATA_AVAILABLE = False
    logger.error(
        "ydata-profiling is not installed. Please install it to use external reporting."
    )


class ExternalReporter:
    """
    Wrapper for external reporting libraries (YData Profiling).
    """

    @staticmethod
    def generate_profile(
        df: pd.DataFrame,
        output_dir: str,
        filename: str = "profile_report.html",
        title: str = "Data Profile",
        time_col: Optional[str] = None,
        block_col: Optional[str] = None,
        minimal: bool = True,
    ) -> Optional[str]:
        """
        Generates a YData Profile Report.

        Args:
            df: DataFrame to profile.
            output_dir: Directory to save the report.
            filename: Name of the output HTML file.
            title: Title of the report.
            time_col: Column name representing time. If provided, enables Time Series mode.
            block_col: Column name representing blocks. Used as time proxy if time_col is missing.
            minimal: If True, generates a minimal report (faster, no correlations).

        Returns:
            Absolute path to the generated report, or None if generation failed.
        """
        if not YDATA_AVAILABLE:
            return None

        try:
            # Determine Time Series Mode
            tsmode = False
            sortby = None

            # Logic: Prefer time_col, fallback to block_col for sequence
            if time_col and time_col in df.columns:
                tsmode = True
                sortby = time_col
                logger.info(f"Time Series Mode ENABLED (using time_col: {time_col})")
            elif block_col and block_col in df.columns:
                tsmode = True
                sortby = block_col
                logger.info(
                    f"Time Series Mode ENABLED (using block_col: {block_col} as sequence proxy)"
                )

            # Prepare output path
            output_path = os.path.join(output_dir, filename)

            # Generate Report
            profile = ProfileReport(
                df,
                title=title,
                tsmode=tsmode,
                sortby=sortby,
                html={"style": {"full_width": True}},
                progress_bar=False,
                minimal=minimal,
            )

            profile.to_file(output_path)

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate YData Profile: {e}")
            return None

    @staticmethod
    def generate_comparison(
        ref_df: pd.DataFrame,
        curr_df: pd.DataFrame,
        output_dir: str,
        filename: str = "comparison_report.html",
        ref_name: str = "Reference (Real)",
        curr_name: str = "Current (Synthetic)",
        time_col: Optional[str] = None,
        block_col: Optional[str] = None,
        minimal: bool = True,
    ) -> Optional[str]:
        """
        Generates a YData Profiling Comparison Report (Two Datasets).

        Args:
            ref_df: Reference DataFrame (e.g., original/real data).
            curr_df: Current DataFrame (e.g., synthetic/drifted data).
            output_dir: Directory to save the report.
            filename: Name of the output HTML file.
            ref_name: Label for the reference dataset.
            curr_name: Label for the current dataset.
            time_col: Optional time column for time-series mode.
            block_col: Optional block column as time proxy.

        Returns:
            Absolute path to the generated report, or None.
        """
        if not YDATA_AVAILABLE:
            return None

        try:
            # Determine Time Series Mode for Comparison
            tsmode = False
            sortby = None

            if time_col and time_col in ref_df.columns and time_col in curr_df.columns:
                tsmode = True
                sortby = time_col
                logger.info(
                    f"Comparison Time Series Mode ENABLED (using time_col: {time_col})"
                )
            elif (
                block_col
                and block_col in ref_df.columns
                and block_col in curr_df.columns
            ):
                tsmode = True
                sortby = block_col
                logger.info(
                    f"Comparison Time Series Mode ENABLED (using block_col: {block_col})"
                )

            # NOTE: usage of tsmode in compare() requires columns to be present.
            # We do NOT drop them if we are in tsmode.
            cols_to_exclude = []
            if not tsmode:
                # Only exclude if NOT in time series mode (to avoid confusion in static comparison)
                if block_col and block_col in ref_df.columns:
                    cols_to_exclude.append(block_col)
                if time_col and time_col in ref_df.columns:
                    cols_to_exclude.append(time_col)

            ref_df_clean = ref_df.drop(columns=cols_to_exclude, errors="ignore")
            curr_df_clean = curr_df.drop(columns=cols_to_exclude, errors="ignore")

            # Generate individual profiles
            ref_report = ProfileReport(
                ref_df_clean,
                title=ref_name,
                tsmode=tsmode,
                sortby=sortby,
                html={"style": {"full_width": True}},
                progress_bar=False,
                minimal=minimal,
            )

            curr_report = ProfileReport(
                curr_df_clean,
                title=curr_name,
                tsmode=tsmode,
                sortby=sortby,
                html={"style": {"full_width": True}},
                progress_bar=False,
                minimal=minimal,
            )

            # Compare the two reports
            comparison_report = ref_report.compare(curr_report)

            # Save to file
            output_path = os.path.join(output_dir, filename)
            comparison_report.to_file(output_path)

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate YData Comparison: {e}")
            return None
