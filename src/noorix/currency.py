"""Pakistani banknote denomination estimation interface.

This module intentionally contains no fabricated classifier. A trained and
validated model must be integrated before returning denomination estimates.
"""


class CurrencyEstimator:
    def predict(self, image):
        """Estimate a denomination using a future validated model."""
        raise NotImplementedError("Currency model is not integrated yet")
