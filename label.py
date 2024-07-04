from datasets import load_dataset
from pii_cloud_services import AzurePIIService, GCPPIIService
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv
import os
import sys
import json

load_dotenv()

def process_text_with_service(service: AzurePIIService | GCPPIIService, text: str) -> Dict[str, List[Dict] | str]:
    extracted_pii = service.extract_pii(text)
    redacted_text = service.redact_pii(text)
    return {"extracted": extracted_pii, "redacted": redacted_text}

def apply_services_to_dataset(dataset, azure_service: AzurePIIService, gcp_service: GCPPIIService):
    def process_row(row):
        azure_results = process_text_with_service(azure_service, row['generated_text'])
        gcp_results = process_text_with_service(gcp_service, row['generated_text'])
        
        row['azure_extracted'] = azure_results['extracted']
        row['azure_redacted'] = azure_results['redacted']
        row['gcp_extracted'] = gcp_results['extracted']
        row['gcp_redacted'] = gcp_results['redacted']
        
        return row

    return dataset.map(process_row, desc="Processing rows")


if __name__ == "__main__":
    # Initialize services
    azure_service = AzurePIIService()
    gcp_service = GCPPIIService()

    hf_username = os.getenv("HF_USERNAME")
    if not hf_username:
        raise ValueError("HF_USERNAME must be set in .env file")

    # Load the dataset
    ds = load_dataset("gretelai/synthetic_pii_finance_multilingual")


    # Define the filter conditions
    document_types = ["Email", "IT support ticket", "Customer support conversational log"]
    pii_types = ["company", "email", "date", "street_address", "name"]
    language = "English"


    filtered_ds = ds['train'].filter(
        lambda example: (
            example['document_type'] in document_types and
            example['language'] == language
        )
    )

    MAX_RECORDS=2500
    filtered_size = len(filtered_ds)
    original_size = len(ds['train'])
    print(f"Original dataset size: {original_size}")
    print(f"Filtered dataset size: {filtered_size}")

    if filtered_size > MAX_RECORDS:
        print(f"\nError: The filtered dataset ({filtered_size} records) exceeds the maximum limit of {MAX_RECORDS} records.")
        print("Please adjust your filters to reduce the number of records or increase the MAX_RECORDS limit")
        sys.exit(1)

    user_input = input(f"\nDo you want to proceed with processing {filtered_size} records? (yes/no): ").lower()
    if user_input != 'yes':
        print("Processing cancelled by user.")
        sys.exit(0)


    # Process the filtered dataset
    processed_dataset = apply_services_to_dataset(filtered_ds, azure_service, gcp_service)

    # Push the processed dataset to Hugging Face Hub
    processed_dataset.push_to_hub(f"{hf_username}/synthetic_pii_finance_multilingual_processed_filtered")

    print("Processed dataset has been pushed to Hugging Face Hub.")