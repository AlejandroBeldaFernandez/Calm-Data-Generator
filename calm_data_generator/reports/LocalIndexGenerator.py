"""
Local Index Generator
Generates a static index.html dashboard    - Uses synthetic data generation (e.g. via Synthcity) to augment the dataset.
"""

import os
import time
import logging

logger = logging.getLogger("LocalIndexGenerator")


class LocalIndexGenerator:
    """
    Generates a local index.html to act as a dashboard for reports.
    """

    HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CalmOps Data Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #f8f9fa; margin: 0; padding: 0; }
        .container { display: flex; height: 100vh; }
        .sidebar { width: 280px; background-color: #343a40; color: white; padding: 20px; display: flex; flex-direction: column; overflow-y: auto; }
        .sidebar h2 { margin-top: 0; font-size: 1.2rem; margin-bottom: 20px; border-bottom: 1px solid #4b545c; padding-bottom: 10px; }
        .section-title { font-size: 0.75rem; text-transform: uppercase; color: #6c757d; margin: 15px 0 8px 0; letter-spacing: 0.5px; }
        .nav-btn { background: none; border: none; color: #c2c7d0; padding: 10px 15px; text-align: left; cursor: pointer; font-size: 0.95rem; border-radius: 4px; margin-bottom: 3px; transition: background 0.2s; width: 100%; }
        .nav-btn:hover { background-color: #495057; color: white; }
        .nav-btn.active { background-color: #007bff; color: white; }
        .content { flex-grow: 1; padding: 0; background-color: white; overflow: hidden; position: relative; }
        iframe { width: 100%; height: 100%; border: none; display: none; }
        iframe.active { display: block; }
        .tab-row { display: flex; align-items: center; margin-bottom: 3px; }
        .tab-row a { color: #6c757d; margin-left: 8px; text-decoration: none; font-size: 1.1rem; }
        .tab-row a:hover { color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>CalmOps Report</h2>
            <!-- YDATA_SECTION -->
            <!-- PLOTLY_SECTION -->
            <!-- SC_SECTION -->
        </div>
        
        <div class="content">
            <!-- IFRAMES_PLACEHOLDER -->
        </div>
    </div>

    <script>
        function showTab(id) {
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            const btn = document.getElementById('btn-' + id);
            if(btn) btn.classList.add('active');

            document.querySelectorAll('iframe').forEach(f => f.classList.remove('active'));
            
            const el = document.getElementById(id);
            if(el) el.classList.add('active');
        }

        document.addEventListener('DOMContentLoaded', () => {
             const priority = ['comparison', 'profile', 'quality_scores', 'density', 'dimensionality', 'quality_evolution'];
             for (const id of priority) {
                 if (document.getElementById(id)) {
                     showTab(id);
                     break;
                 }
             }
        });
    </script>
</body>
</html>
"""

    @staticmethod
    def create_index(output_dir: str) -> str:
        """
        Scans output_dir for artifacts and generates index.html.
        """
        try:
            ts = int(time.time())

            # === YData Reports Section ===
            ydata_section = ""
            iframes_html = ""

            ydata_reports = {
                "comparison": {
                    "files": ["comparison_report.html"],
                    "label": "YData Comparison",
                },
                "profile": {
                    "files": ["generated_profile.html", "profile_report.html"],
                    "label": "Generated Data Profile",
                },
            }

            found_ydata = []
            for rid, config in ydata_reports.items():
                for fname in config["files"]:
                    if os.path.exists(os.path.join(output_dir, fname)):
                        found_ydata.append((rid, fname, config["label"]))
                        break

            if found_ydata:
                ydata_section = '<div class="section-title">YData Reports</div>\n'
                for rid, fname, label in found_ydata:
                    ydata_section += f'''
                    <div class="tab-row">
                        <button class="nav-btn" onclick="showTab('{rid}')" id="btn-{rid}">{label}</button>
                        <a href="{fname}" target="_blank" title="Open in New Tab">-></a>
                    </div>
                    '''
                    iframes_html += f'<iframe id="{rid}" src="{fname}?v={ts}" scrolling="yes"></iframe>\n'

            # === Plotly Reports Section ===
            plotly_section = ""

            plotly_reports = {
                "quality_scores": {
                    "file": "quality_scores.html",
                    "label": "Quality Scores",
                },
                "quality_evolution": {
                    "file": "quality_evolution.html",
                    "label": "Quality Evolution",
                },
                "drift_stats": {
                    "file": "drift_stats.html",
                    "label": "Drift Statistics",
                },
                "evolution_plot": {
                    "file": "evolution_plot.html",
                    "label": "Feature Evolution (ScenarioInjector)",
                },
                "plot_comparison": {
                    "file": "plot_comparison.html",
                    "label": "Distribution Comparison",
                },
                "density": {
                    "file": "density_plots.html",
                    "label": "Density Plots",
                },
                "dimensionality": {
                    "file": "dimensionality_plot.html",
                    "label": "PCA Visualization",
                },
                "discriminator_metrics": {
                    "file": "discriminator_metrics.html",
                    "label": "Adversarial Validation",
                },
                "discriminator_explainability": {
                    "file": "discriminator_explainability.html",
                    "label": "Discriminator Explainability",
                },
            }

            found_plotly = []
            for rid, config in plotly_reports.items():
                fname = config["file"]
                if os.path.exists(os.path.join(output_dir, fname)):
                    found_plotly.append((rid, fname, config["label"]))

            if found_plotly:
                plotly_section = '<div class="section-title">Interactive Plots</div>\n'
                for rid, fname, label in found_plotly:
                    plotly_section += f'''
                    <div class="tab-row">
                        <button class="nav-btn" onclick="showTab('{rid}')" id="btn-{rid}">{label}</button>
                        <a href="{fname}" target="_blank" title="Open in New Tab">-></a>
                    </div>
                    '''
                    iframes_html += f'<iframe id="{rid}" src="{fname}?v={ts}" scrolling="yes"></iframe>\n'

            # === Single Cell Reports Section ===
            sc_section = ""
            sc_reports = {
                "scgft": {
                    "file": "scgft_report.html",
                    "label": "scGFT Evaluation",
                },
            }
            
            found_sc = []
            for rid, config in sc_reports.items():
                fname = config["file"]
                if os.path.exists(os.path.join(output_dir, fname)):
                    found_sc.append((rid, fname, config["label"]))
            
            if found_sc:
                sc_section = '<div class="section-title">Single Cell Analysis</div>\n'
                for rid, fname, label in found_sc:
                    sc_section += f'''
                    <div class="tab-row">
                        <button class="nav-btn" onclick="showTab('{rid}')" id="btn-{rid}">{label}</button>
                        <a href="{fname}" target="_blank" title="Open in New Tab">-></a>
                    </div>
                    '''
                    iframes_html += f'<iframe id="{rid}" src="{fname}?v={ts}" scrolling="yes"></iframe>\n'

            # === Assemble HTML ===
            html = LocalIndexGenerator.HTML_TEMPLATE
            html = html.replace("<!-- YDATA_SECTION -->", ydata_section)
            html = html.replace("<!-- PLOTLY_SECTION -->", plotly_section)
            html = html.replace("<!-- SC_SECTION -->", sc_section)
            html = html.replace("<!-- IFRAMES_PLACEHOLDER -->", iframes_html)

            index_path = os.path.join(output_dir, "index.html")
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(html)

            logger.info(f"Dashboard created at: {index_path}")
            return index_path

        except Exception as e:
            logger.error(f"Failed to create dashboard index: {e}")
            return ""
