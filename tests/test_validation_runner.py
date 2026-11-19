from scripts.run_full_validation import validate_notebook_document


def test_valid_empty_notebook_is_accepted():
    assert validate_notebook_document({
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [],
    }) is None


def test_notebook_requires_object_root():
    assert validate_notebook_document([]) == "top-level JSON value must be an object"


def test_notebook_rejects_unsupported_format():
    assert validate_notebook_document({"nbformat": 3, "nbformat_minor": 0, "cells": []}) == "nbformat must be 4"


def test_notebook_rejects_invalid_cell_source():
    document = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [{"cell_type": "markdown", "source": 42}],
    }
    assert validate_notebook_document(document) == "cell 0 source must be a string or list"


def test_code_cell_requires_list_outputs():
    document = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [{"cell_type": "code", "source": "print('hi')", "outputs": {}}],
    }
    assert validate_notebook_document(document) == "code cell 0 outputs must be a list"
