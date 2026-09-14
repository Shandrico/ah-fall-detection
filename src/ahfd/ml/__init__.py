"""Optional learned models (scikit-learn), layered on the rule-based system.

These never replace the interpretable rule-based detector; they are an optional
re-ranker trained on labelled clips. scikit-learn is an optional dependency,
imported lazily inside the functions that need it, so the core install stays lean.
"""
