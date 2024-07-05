from datasets import load_dataset, Dataset
from pii_services import AzurePIIService, GCPPIIService
import os
from dotenv import load_dotenv
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(filename='pii_processing.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Get Hugging Face username from environment variable
hf_username = os.getenv("HUGGINGFACE_USERNAME")
if not hf_username:
    raise ValueError("HUGGINGFACE_USERNAME not set in environment variables")

def process_row(row, azure_service, gcp_service, gcp_info_types):
    try:
        text = row['generated_text']
        azure_result = azure_service.recognize_pii([text])
        gcp_result = gcp_service.recognize_pii([text], gcp_info_types)
        
        row['azure_results'] = str(azure_result)
        row['gcp_results'] = str(gcp_result)
        row['processing_status'] = 'success'
    except Exception as e:
        error_message = f"Error processing row: {str(e)}"
        logging.error(error_message)
        row['azure_results'] = None
        row['gcp_results'] = None
        row['processing_status'] = 'error'
        row['error_message'] = error_message
    return row

def main():
    try:
        # Load dataset
        print("Loading dataset...")
        ds = load_dataset("gretelai/synthetic_pii_finance_multilingual")

        # Apply filters
        document_types = ["Email", "IT support ticket", "Customer support conversational log"]
        language = "English"
        
        filtered_ds = ds['train'].filter(
            lambda example: (
                example['document_type'] in document_types and
                example['language'] == language
            )
        )

        print(f"Filtered dataset size: {len(filtered_ds)}")

        # Initialize services
        azure_service = AzurePIIService()
        gcp_service = GCPPIIService()

        # Define info types for GCP
        gcp_info_types = ["PERSON_NAME", "EMAIL_ADDRESS", "DATE", "STREET_ADDRESS", "ORGANIZATION_NAME"]

        # Process the dataset
        processed_ds = filtered_ds.map(
            lambda row: process_row(row, azure_service, gcp_service, gcp_info_types),
            desc="Processing rows"
        )

        # Print summary
        success_count = sum(1 for row in processed_ds if row['processing_status'] == 'success')
        error_count = sum(1 for row in processed_ds if row['processing_status'] == 'error')
        print(f"Processed {len(processed_ds)} rows. Successes: {success_count}, Errors: {error_count}")

        # Push to Hugging Face Hub
        repo_name = f"{hf_username}/filtered-pii-results"
        processed_ds.push_to_hub(repo_name)

        print(f"Dataset uploaded successfully to https://huggingface.co/datasets/{repo_name}")

    except Exception as e:
        logging.error(f"An error occurred during script execution: {str(e)}")
        print(f"An error occurred. Check the log file for details.")

if __name__ == "__main__":
    main()
