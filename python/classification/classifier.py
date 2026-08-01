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

# Model interpreter cache (prevents reloading model weights on every call)
_INTERPRETER_CACHE = {}


def _get_interpreter(model_name: str):
    """Internal helper to load and cache TFLite interpreters dynamically."""
    if model_name in _INTERPRETER_CACHE:
        return _INTERPRETER_CACHE[model_name]

    if model_name not in USERCONFIG.MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in USERCONFIG.MODEL_REGISTRY.")

    model_info = USERCONFIG.MODEL_REGISTRY[model_name]
    model_path = model_info["model_path"]
    if not os.path.isabs(model_path):
        model_path = os.path.normpath(os.path.join(_HERE, model_path))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    _INTERPRETER_CACHE[model_name] = interpreter
    return interpreter


# =============================================================================
# CHUNK 1: Reshape & Wrap Image
# =============================================================================
def chunk_wrap_and_resize(image_input, target_width: int, target_height: int) -> np.ndarray:
    """Accepts file path or BGR image array and resizes to target model shape."""
    if isinstance(image_input, str):
        frame = cv2.imread(image_input)
        if frame is None:
            raise ValueError(f"Could not load image from path: {image_input}")
    elif isinstance(image_input, np.ndarray):
        frame = image_input
    else:
        raise TypeError("image_input must be a file path string or a numpy ndarray.")

    resized = cv2.resize(frame, (target_width, target_height))
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb_frame


# =============================================================================
# CHUNK 2: Data Normalization
# =============================================================================
def chunk_normalize_data(rgb_frame: np.ndarray, input_details: dict) -> np.ndarray:
    """Converts image datatype (UINT8, INT8, FP32) based on quantized model specs."""
    dtype = input_details['dtype']

    if dtype == np.uint8:
        # Standard UINT8 quantized models [0, 255]
        tensor = np.expand_dims(rgb_frame, axis=0).astype(np.uint8)

    elif dtype == np.int8:
        # INT8 Quantized models [-128, 127]
        scale, zero_point = input_details.get('quantization', (1.0, 0))
        if scale > 0:
            tensor = (rgb_frame / scale) + zero_point
        else:
            tensor = rgb_frame.astype(np.int16) - 128
        tensor = np.expand_dims(tensor, axis=0).astype(np.int8)

    else:
        # Float32 normalized models [0.0, 1.0]
        tensor = np.expand_dims(rgb_frame, axis=0).astype(np.float32) / 255.0

    return tensor


# =============================================================================
# CHUNK 3: Run Inference
# =============================================================================
def chunk_run_inference(interpreter, tensor_data: np.ndarray, labels: list) -> dict:
    """Executes model inference and returns raw detected object and confidence."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], tensor_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Dequantize output if necessary
    if output_details[0]['dtype'] in [np.int8, np.uint8]:
        scale, zero_point = output_details[0]['quantization']
        if scale > 0:
            output_data = scale * (output_data.astype(np.float32) - zero_point)

    # Convert logits to probabilities via Softmax if not normalized
    if not np.isclose(np.sum(output_data), 1.0, atol=1e-2):
        exp_scores = np.exp(output_data - np.max(output_data))
        scores = exp_scores / np.sum(exp_scores)
    else:
        scores = output_data

    top_idx = int(np.argmax(scores))
    confidence = float(scores[top_idx])
    detected_object = labels[top_idx] if top_idx < len(labels) else "unknown"

    return {"object": detected_object, "confidence": confidence}


# =============================================================================
# CHUNK 4: Hash Table Lookup (Location Rules Engine)
# =============================================================================
def chunk_lookup_disposal_class(detected_object: str, confidence: float) -> str or None:
    """Validates confidence against threshold and maps object to standard disposal class."""
    if confidence < USERCONFIG.CONFIDENCE_THRESHOLD:
        return None

    country = USERCONFIG.COUNTRY
    state = USERCONFIG.STATE
    county = USERCONFIG.COUNTY

    try:
        county_map = USERCONFIG.LOCAL_RULES[country][state][county]
        category = county_map.get(detected_object.lower(), None)

        # Enforce strict 3-class system constraint ['Recyclable', 'Non-Recyclable', 'Hazardous']
        if category in ['Recyclable', 'Non-Recyclable', 'Hazardous']:
            return category
        return None

    except KeyError:
        # Location path not configured in hash table
        return None


# =============================================================================
# ENTRY POINT USED BY THE ORCHESTRATOR
# =============================================================================
def classify_raw(image_input, model_name: str = None):
    """Return (label, confidence) and make no disposal decision.

    Chunks 1-3 only. The label -> category step is deliberately skipped: that
    mapping lives in disposal_rules.yaml so disposal policy can change without
    retraining, and so the MCU's categories have a single source of truth.
    """
    selected_model = model_name if model_name else USERCONFIG.DEFAULT_MODEL_NAME

    interpreter = _get_interpreter(selected_model)
    input_details = interpreter.get_input_details()
    labels = USERCONFIG.MODEL_REGISTRY[selected_model]["labels"]

    target_height = input_details[0]['shape'][1]
    target_width = input_details[0]['shape'][2]

    rgb_frame = chunk_wrap_and_resize(image_input, target_width, target_height)
    tensor_data = chunk_normalize_data(rgb_frame, input_details[0])
    result = chunk_run_inference(interpreter, tensor_data, labels)

    return result["object"], result["confidence"]


# =============================================================================
# ORIGINAL ENTRY POINT (kept for standalone use; returns a disposal class)
# =============================================================================
def classify_image(image_input, model_name: str = None) -> str or None:
    """
    Main entry point for external orchestrator.
    
    Inputs:
        image_input: File path (str) OR cv2 BGR image array.
        model_name (optional): String key for pretrained model.
        
    Returns:
        One of ['Recyclable', 'Non-Recyclable', 'Hazardous'] or None.
    """
    selected_model = model_name if model_name else USERCONFIG.DEFAULT_MODEL_NAME
    
    # Load interpreter & configuration metadata
    interpreter = _get_interpreter(selected_model)
    input_details = interpreter.get_input_details()
    labels = USERCONFIG.MODEL_REGISTRY[selected_model]["labels"]

    # Target shape required by chosen model
    target_height = input_details[0]['shape'][1]
    target_width = input_details[0]['shape'][2]

    # Pipeline execution through functional chunks
    # Chunk 1: Reshape & wrap
    rgb_frame = chunk_wrap_and_resize(image_input, target_width, target_height)

    # Chunk 2: Normalize
    tensor_data = chunk_normalize_data(rgb_frame, input_details[0])

    # Chunk 3: Inference
    inference_result = chunk_run_inference(interpreter, tensor_data, labels)

    # Chunk 4: Location-based rule lookup
    final_output = chunk_lookup_disposal_class(
        detected_object=inference_result["object"],
        confidence=inference_result["confidence"]
    )

    return final_output