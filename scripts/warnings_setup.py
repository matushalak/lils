import warnings


def silence_known_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*pkg_resources package is slated for removal.*",
        category=UserWarning,
    )
