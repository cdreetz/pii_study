from datasets import load_dataset

ds = load_dataset("gretelai/synthetic_pii_finance_multilingual")
document_types = ["Email", "IT support ticket", "Customer support conversational log"]
language = "English"

filtered_ds = ds.filter(
    lambda example: (
        example['document_type'] in document_types and
        example['language'] == language
    )
)
