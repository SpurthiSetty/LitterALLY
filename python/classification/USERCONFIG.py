# Model configuration for the classifier.
#
# Disposal policy is deliberately NOT here. label -> category lives in
# python/disposal_rules.yaml, which is the single source of truth, so that
# policy can change without touching code and the MCU has one table to match.

# Fallback default model if the caller does not specify one
DEFAULT_MODEL_NAME = "mobilenet_imagenet"

# How many frames a single classification is built from, and how their
# predictions are combined. See classifier.classify_burst.
BURST_FRAMES = 3
ENSEMBLE_STRATEGY = "mean"  # "mean" | "max" | "vote"

# Model registry: alias -> local path and the label list for that model.
#
# Label ORDER matters and is not cosmetic: argmax returns an index, and the
# label is looked up by position. A reordered list silently mislabels every
# prediction. These match the labels.txt shipped alongside each model.
MODEL_REGISTRY = {
    # Stock MobileNetV2 on ImageNet, int8 quantized. A thousand general classes
    # rather than a dozen waste ones, which suits an architecture that maps
    # label -> category outside the model. It names what it actually sees, so
    # an empty scene comes back as "stove" with low confidence instead of a
    # waste-only model's confident guess at which kind of rubbish a kitchen is.
    #
    # Labels come from a file: a thousand of them do not belong inline. Only
    # the text before the first comma is used, so "notebook, notebook computer"
    # becomes "notebook".
    "mobilenet_imagenet": {
        "model_path": "./models/mobilenet_imagenet.tflite",
        "labels_file": "./models/imagenet_labels.txt",
    },
    "wastenet_mobilenetv2": {
        "model_path": "./models/wastenet_model.tflite",
        "labels": ["battery", "biological", "brown-glass", "cardboard", "clothes",
                   "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"]
    },
    "eco_ai_mobilenet": {
        "model_path": "./models/eco_ai_model.tflite",
        "labels": ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    }
}
