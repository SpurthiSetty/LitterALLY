# USERCONFIG.py

# Location parameters
COUNTRY = "USA"
STATE = "California"
COUNTY = "San Diego"

# Fallback default model if orchestrator does not specify one
DEFAULT_MODEL_NAME = "wastenet_mobilenetv2"

# Global Confidence Threshold (Predictions below this return None)
CONFIDENCE_THRESHOLD = 0.65

# Model registry: Maps alias to local path and labels file
MODEL_REGISTRY = {
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

# Location-based Disposal Rules Lookup Table
# Structure: LOCAL_RULES[COUNTRY][STATE][COUNTY][OBJECT_LABEL] -> Disposal Class
LOCAL_RULES = {
    "USA": {
        "California": {
            "San Diego": {
                "cardboard": "Recyclable",
                "paper": "Recyclable",
                "plastic": "Recyclable",
                "metal": "Recyclable",
                "glass": "Recyclable",
                "brown-glass": "Recyclable",
                "green-glass": "Recyclable",
                "white-glass": "Recyclable",
                "battery": "Hazardous",
                "e_waste": "Hazardous",
                "biological": "Non-Recyclable",
                "clothes": "Non-Recyclable",
                "shoes": "Non-Recyclable",
                "trash": "Non-Recyclable"
            }
        }
    }
}