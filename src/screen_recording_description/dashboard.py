"""Combined Streamlit app to display both Results and Service Metrics in one launch.

    streamlit run src/screen_recording_description/dashboard.py
"""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Screen Recording Description",
    layout="wide",
)

_here = Path(__file__).parent
page = st.navigation([
    st.Page(_here / "results_dashboard.py", title="Evaluation Results", icon="📊", default=True),
    st.Page(_here / "service_dashboard.py", title="Service Metrics", icon="📈"),
])
page.run()
