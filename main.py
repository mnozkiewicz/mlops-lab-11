from src.scripts.download import download_s3_folder
from src.scripts.export_classifier_to_onnx import export_classifier_to_onnx
from src.scripts.export_sentence_transformer_to_onnx import export_model_to_onnx
from src.scripts.settings import Settings


def main():
    settings = Settings()
    download_s3_folder(settings)
    export_classifier_to_onnx(settings)
    export_model_to_onnx(settings)

if __name__ == "__main__":
    main()
