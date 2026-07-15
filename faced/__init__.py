"""faced — an instrument panel / face for a language model's internal emotion concepts.

Reads linear "emotion" directions off a model's residual stream (via HuggingFace
transformers forward hooks / hidden states), calibrates them into 0-100% meters,
renders them as a live face, and can steer them. Model-agnostic: backends live in
config/models.yaml.
"""

__version__ = "0.1.0"
