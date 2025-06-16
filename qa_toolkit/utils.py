# qa_toolkit/utils.py
import os
import base64
import logging
from io import BytesIO
from typing import Optional
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def save_plot_and_get_uri(fig: plt.Figure, 
                          filename_stem: str, 
                          output_dir: str, 
                          plot_img_subdir_name: str,
                          embed_plots: bool,
                          save_individual: bool) -> Optional[str]:
    """
    Saves a matplotlib figure and returns its data URI for embedding or file path.
    Args:
        fig: Matplotlib figure object.
        filename_stem: Base name for the plot file (e.g., "length_distribution").
        output_dir: Main output directory for reports.
        plot_img_subdir_name: Name of the subdirectory within output_dir for plots.
        embed_plots: If True, returns a base64 data URI. Otherwise, returns relative path.
        save_individual: If True, always saves the plot as a separate file.
    Returns:
        A string (data URI or relative file path) or None if saving/embedding fails.
    """
    plot_uri_or_path: Optional[str] = None
    plot_saved_to_file = False
    
    plot_img_full_dir = os.path.join(output_dir, plot_img_subdir_name)
    
    if save_individual:
        os.makedirs(plot_img_full_dir, exist_ok=True)
        file_path = os.path.join(plot_img_full_dir, f"{filename_stem}.png")
        try:
            fig.savefig(file_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {file_path}")
            plot_saved_to_file = True
            if not embed_plots: # If not embedding, use this relative path for linking
                plot_uri_or_path = os.path.join(plot_img_subdir_name, f"{filename_stem}.png")
        except Exception as e:
            logger.error(f"Failed to save plot {filename_stem} to file: {e}")

    if embed_plots:
        try:
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches='tight') # Lower DPI for embedding
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plot_uri_or_path = f"data:image/png;base64,{img_str}"
        except Exception as e:
            logger.error(f"Failed to generate base64 for plot {filename_stem}: {e}")
            # Fallback to path if embedding failed but file was saved and not already set
            if plot_saved_to_file and plot_uri_or_path is None:
                 plot_uri_or_path = os.path.join(plot_img_subdir_name, f"{filename_stem}.png")
    
    plt.close(fig) # Close figure to free memory
    return plot_uri_or_path