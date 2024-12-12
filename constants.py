from chromadb.config import Settings

CHROMA_SETTINGS = Settings(
    # chroma_implementation='duckdb+parquet',  # Corrected spelling here
    persist_directory='./db',
    anonymized_telemetry=False,
    is_persistent=True,
)
