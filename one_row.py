from datasets import load_dataset, Dataset
from pii_services import AzurePIIService, GCPPIIService
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get Hugging Face username from environment variable
hf_username = os.getenv("HUGGINGFACE_USERNAME")
if not hf_username:
    raise ValueError("HUGGINGFACE_USERNAME not set in environment variables")

# Load dataset
print("Loading dataset...")
ds = load_dataset("gretelai/synthetic_pii_finance_multilingual")

# Select a single row
single_row = ds['train'][0]
single_row_ds = Dataset.from_dict({k: [v] for k, v in single_row.items()})

# Initialize services
azure_service = AzurePIIService()
gcp_service = GCPPIIService()

# Define info types for GCP
gcp_info_types = ["PERSON_NAME", "EMAIL_ADDRESS", "DATE", "STREET_ADDRESS", "ORGANIZATION_NAME"]

# Process the single row
def process_row(row):
    text = row['generated_text']
    azure_result = azure_service.recognize_pii([text])
    gcp_result = gcp_service.recognize_pii([text], gcp_info_types)
    
    row['azure_results'] = str(azure_result)
    row['gcp_results'] = str(gcp_result)
    return row

processed_ds = single_row_ds.map(process_row)

# Print the processed row
print("Processed row:")
print(processed_ds[0])

# Push to Hugging Face Hub
repo_name = f"{hf_username}/single-row-pii-test"
processed_ds.push_to_hub(repo_name)

print(f"Dataset uploaded successfully to https://huggingface.co/datasets/{repo_name}")
