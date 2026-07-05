from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep the real Streamlit app in app.py, but support:
# streamlit run frontend/streamlit_app.py
from app import main


if __name__ == "__main__":
    main()
