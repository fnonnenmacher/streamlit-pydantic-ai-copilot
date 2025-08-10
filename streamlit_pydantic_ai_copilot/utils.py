from streamlit.runtime.scriptrunner import get_script_run_ctx


def to_kebab_case(input: str) -> str:
    return "".join("-" + c.lower() if c.isupper() and idx > 0 else c.lower() for idx, c in enumerate(input))


def streamlit_session_id() -> str:
    # Hack, uses internal API to determine the session id, might break in the future
    ctx = get_script_run_ctx()
    assert ctx is not None
    return ctx.session_id
