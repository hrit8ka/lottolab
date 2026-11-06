from streamlit.testing.v1 import AppTest


def test_app_runs_all_pages():
    at = AppTest.from_file("../dashboard/app.py", default_timeout=120).run()
    assert not at.exception
    for page in ["Number explorer", "Prediction lab", "Scoreboard"]:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, at.exception)
