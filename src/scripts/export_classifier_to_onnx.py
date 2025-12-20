import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from pathlib import Path

from settings import Settings


def export_classifier_to_onnx(settings: Settings):
    print(f"Loading classifier from {settings.classifier_joblib_path}...")
    classifier = joblib.load(settings.classifier_joblib_path)

    # define input shape: (batch_size, embedding_dim)
    initial_type = [("float_input", FloatTensorType([None, settings.embedding_dim]))]

    print("Converting to ONNX...")
    onnx_model = convert_sklearn(
        model=classifier,
        initial_types=initial_type
    )
    
    output_path = Path(settings.onnx_classifier_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving ONNX model to {output_path}...")
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


if __name__ == '__main__':
    settings = Settings()
    export_classifier_to_onnx(settings)