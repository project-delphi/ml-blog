# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
# ---

# %%
"""placeholder module with single function.

Use this module as a template to work with google convention for pydocstyle.
"""


# %%
def hello(name: str = "Ravi") -> str:
    """Say hi to name.

    Args:
        name (str, optional): _description_. Defaults to "Ravi".

    Returns
    -------
        str: _description_
    """
    return f"Hi {name}"
