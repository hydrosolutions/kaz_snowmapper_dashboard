import panel as pn
from datetime import datetime, timedelta
import pytz
import logging

logger = logging.getLogger(__name__)

def check_data_freshness(time_bounds, config):
    """
    Check if the data is fresh enough based on time bounds

    Parameters:
    -----------
    time_bounds : tuple
        Tuple of (start_time, end_time) integers representing days offset
    config : dict
        Configuration dictionary containing dashboard settings

    Returns:
    --------
    bool
        True if data is stale, False otherwise
    """
    if not time_bounds or len(time_bounds) != 2:
        logger.warning("Invalid time bounds structure")
        return True

    try:
        end_time = int(time_bounds[1])
        day_slider_min = int(config['dashboard']['day_slider_min'])

        logger.debug(f"End time offset: {end_time}")
        logger.debug(f"Day slider min: {day_slider_min}")

        # Data is stale if the end time offset is less than the minimum allowed
        is_stale = end_time < day_slider_min

        logger.debug(f"Is data stale? {is_stale}")
        return is_stale

    except Exception as e:
        logger.error(f"Error checking data freshness: {str(e)}")
        return True

def create_warning_component():
    """
    Create a warning component that can be shown/hidden based on data freshness

    Returns:
    --------
    panel.Column
        A panel component containing the warning message
    """
    warning_msg = pn.pane.Alert(
        """⚠️ Warning: No recent snow data available. This could be due to:
        - Temporary data processing issues
        - Network connectivity problems
        - Server maintenance

        Please check back later or contact support if this persists.""",
        alert_type='danger',
        sizing_mode='stretch_width',
        height=100,  # Explicit height to ensure visibility
        margin=(10, 0, 40, 0)  # Add some vertical margin
    )

    return warning_msg

class DataFreshnessManager:
    """
    Manages the display of data freshness warnings in the dashboard
    """
    def __init__(self):
        self.warning_component = create_warning_component()
        self.warning_component.visible = False
        logger.debug("DataFreshnessManager initialized")

    def set_warning_visibility(self, is_visible):
        """
        Set the visibility of the warning component

        Parameters:
        -----------
        is_visible : bool
            True to show the warning, False to hide it
        """
        self.warning_component.visible = is_visible
        logger.debug(f"Warning visibility set to: {is_visible}")
        print(f"Warning visibility set to: {is_visible}")

    def update_warning_visibility(self, time_bounds, config):
        """
        Update the visibility of the warning based on data freshness

        Parameters:
        -----------
        time_bounds : tuple
            Tuple of (start_time, end_time) integers representing days offset
        config : dict
            Configuration dictionary containing dashboard settings
        """
        try:
            logger.debug(f"Updating warning visibility with time_bounds: {time_bounds}")
            print((f"Updating warning visibility with time_bounds: {time_bounds}"))
            is_stale = check_data_freshness(time_bounds, config)
            self.warning_component.visible = is_stale
            logger.debug(f"Warning visibility set to: {is_stale}")
            print((f"Warning visibility set to: {is_stale}"))
        except Exception as e:
            logger.error(f"Error updating warning visibility: {str(e)}")
            self.warning_component.visible = True

    def get_warning_component(self):
        """
        Get the warning component for integration into the dashboard

        Returns:
        --------
        panel.Column
            The warning component
        """
        return self.warning_component