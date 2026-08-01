import os

import cv2
import numpy as np

# ai-edge-litert is Google's continuation of tflite_runtime and runs the same
# .tflite files with the same Interpreter API. It is used here because
# tflite-runtime publishes no wheel for this board: Python 3.13 on aarch64
# resolves to "No matching distribution found".
from ai_edge_litert import interpreter as tflite

try:
    from . import USERCONFIG
except ImportError:  # still importable as a standalone script
    import USERCONFIG

# Model paths in the registry are relative to this package, not to whatever
# directory the app happens to be launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))

# Loading weights costs far more than inference, so interpreters are cached.
_INTERPRETER_CACHE = {}

# Where per-frame output goes. The orchestrator swaps this for a sink that also
# forwards to the MCU serial monitor; the Bridge stays out of this module.
_log = print


def set_logger(fn):
    global _log
    _log = fn


_LABEL_CACHE = {}


def labels_for(model_name: str):
    """Label list for a model, inline or from a file.

    A thousand ImageNet labels do not belong in a config literal, and the file
    ships beside the model. Only the text before the first comma is kept, so
    "notebook, notebook computer" is referred to as "notebook".
    """
    if model_name in _LABEL_CACHE:
        return _LABEL_CACHE[model_name]

    info = USERCONFIG.MODEL_REGISTRY[model_name]
    if "labels" in info:
        labels = list(info["labels"])
    else:
        path = info["labels_file"]
        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(_HERE, path))
        with open(path) as handle:
            labels = [line.split(",")[0].strip() for line in handle if line.strip()]

    _LABEL_CACHE[model_name] = labels
    return labels


def _get_interpreter(model_name: str):
    if model_name in _INTERPRETER_CACHE:
        return _INTERPRETER_CACHE[model_name]

    if model_name not in USERCONFIG.MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in USERCONFIG.MODEL_REGISTRY.")

    model_path = USERCONFIG.MODEL_REGISTRY[model_name]["model_path"]
    if not os.path.isabs(model_path):
        model_path = os.path.normpath(os.path.join(_HERE, model_path))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    _INTERPRETER_CACHE[model_name] = interpreter
    return interpreter


def _wrap_and_resize(image_input, target_width: int, target_height: int) -> np.ndarray:
    """Accept a file path or a BGR array; return an RGB frame at the model's size."""
    if isinstance(image_input, str):
        frame = cv2.imread(image_input)
        if frame is None:
            raise ValueError(f"Could not load image from path: {image_input}")
    elif isinstance(image_input, np.ndarray):
        frame = image_input
    else:
        raise TypeError("image_input must be a file path string or a numpy ndarray.")

    resized = cv2.resize(frame, (target_width, target_height))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def _normalize(rgb_frame: np.ndarray, input_details: dict) -> np.ndarray:
    """Match the tensor dtype the model was quantized for."""
    dtype = input_details['dtype']

    if dtype == np.uint8:
        return np.expand_dims(rgb_frame, axis=0).astype(np.uint8)

    if dtype == np.int8:
        scale, zero_point = input_details.get('quantization', (1.0, 0))
        if scale > 0:
            tensor = (rgb_frame / scale) + zero_point
        else:
            tensor = rgb_frame.astype(np.int16) - 128
        return np.expand_dims(tensor, axis=0).astype(np.int8)

    return np.expand_dims(rgb_frame, axis=0).astype(np.float32) / 255.0


def _probabilities(interpreter, tensor_data: np.ndarray) -> np.ndarray:
    """Run the model and return a probability vector over the label set."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], tensor_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]

    if output_details[0]['dtype'] in (np.int8, np.uint8):
        scale, zero_point = output_details[0]['quantization']
        if scale > 0:
            output = scale * (output.astype(np.float32) - zero_point)

        # A quantized classifier emits a distribution already, so it must not be
        # softmaxed again. Classes below the quantization step round to zero, so
        # on an unremarkable image the whole vector can be zeros - and softmax
        # turns that into a uniform 1/N, reporting 0.001 confidence for every
        # class instead of admitting it recognised nothing. Renormalise instead,
        # and let an all-zero output stay zero.
        total = float(output.sum())
        return output / total if total > 0 else output.astype(np.float32)

    # Float outputs may be logits; softmax only if they are not a distribution.
    if not np.isclose(np.sum(output), 1.0, atol=1e-2):
        exp_scores = np.exp(output - np.max(output))
        return exp_scores / np.sum(exp_scores)

    return output.astype(np.float32)


def _combine(prob_vectors, strategy: str) -> np.ndarray:
    """Reduce several per-frame probability vectors to one.

    mean - average the distributions. The default: a label has to do well
           across frames rather than get lucky on one, so a single blurred or
           badly lit frame cannot carry the result.
    max  - keep the vector whose top-1 was most confident. Useful when most
           frames are poor but one is clean; also the most easily fooled,
           since one confidently wrong frame wins outright.
    vote - majority of per-frame argmax, scored by that label's mean
           probability. Most robust to outliers, least informative when every
           frame disagrees.
    """
    stacked = np.vstack(prob_vectors)

    if strategy == "max":
        return stacked[int(np.argmax(stacked.max(axis=1)))]

    if strategy == "vote":
        winners = stacked.argmax(axis=1)
        counts = np.bincount(winners, minlength=stacked.shape[1])
        chosen = int(np.argmax(counts))
        combined = np.zeros(stacked.shape[1], dtype=np.float32)
        combined[chosen] = float(stacked[:, chosen].mean())
        return combined

    return stacked.mean(axis=0)


def classify_burst(frames, model_name: str = None, strategy: str = None):
    """Classify several frames of the same item and return (label, confidence).

    No disposal decision is made here. label -> category lives in
    disposal_rules.yaml so policy can change without retraining.
    """
    if not frames:
        return "unknown", 0.0

    model = model_name or USERCONFIG.DEFAULT_MODEL_NAME
    strategy = strategy or USERCONFIG.ENSEMBLE_STRATEGY

    interpreter = _get_interpreter(model)
    input_details = interpreter.get_input_details()[0]
    labels = labels_for(model)

    target_height = input_details['shape'][1]
    target_width = input_details['shape'][2]

    prob_vectors = [
        _probabilities(interpreter, _normalize(
            _wrap_and_resize(frame, target_width, target_height), input_details))
        for frame in frames
    ]

    # Per-frame verdicts, so a combined result that looks odd can be traced back
    # to whether the frames disagreed or were uniformly weak.
    for i, vector in enumerate(prob_vectors, start=1):
        idx = int(np.argmax(vector))
        name = labels[idx] if idx < len(labels) else "unknown"
        _log(f"[vision]   frame {i}/{len(prob_vectors)}: {name} {float(vector[idx]):.3f}")

    scores = _combine(prob_vectors, strategy)
    top = int(np.argmax(scores))

    label = labels[top] if top < len(labels) else "unknown"
    confidence = float(scores[top])
    _log(f"[vision]   {strategy} -> {label} {confidence:.3f}")

    return label, confidence


def classify_raw(image_input, model_name: str = None):
    """Single-frame convenience wrapper around classify_burst."""
    return classify_burst([image_input], model_name=model_name)
