
import tensorflow as tf
import os

models = [
    "DenseNet121_oral_diseases.keras",
    "EfficientNetB0_oral_diseases.keras",
    "MobileNetV2_oral_diseases.keras"
]

for name in models:
    print(f"\nTesting: {name}")

    path = os.path.join("models", name)

    try:
        model = tf.keras.models.load_model(
            path,
            compile=False
        )

        print("SUCCESS")
        print("Input shape:", model.input_shape)
        print("Output shape:", model.output_shape)

    except Exception as e:
        print("FAILED:", str(e))
