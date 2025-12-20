from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    s3_bucket_name: str
    s3_model_path: str

    artifacts_path: str
    model_folder_path: str

    classifier_joblib_path: str
    sentence_transformer_dir: str
    embedding_dim: int

    onnx_classifier_path: str
    onnx_embedding_model_path: str
    tokenizer_path: str


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )