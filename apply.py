import os
import sys
import time
from tqdm import tqdm
from typing import List
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, HfFolder
from pii_services import AzurePIIService, GCPPIIService
from dotenv import load_dotenv
import tenacity

load_dotenv()

CHECKPOINT_INTERVAL = 100

@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=4, max=10))
def process_text_with_retry(service, text, info_types=None):
    if info_types:
        return service.recognize_pii([text], info_types)
    else:
        return service.recognize_pii([text])

def save_checkpoint(processed_rows, filename='checkpoint.json'):
    with open(filename, 'w') as f:
        json.dump(processed_rows, f)

def load_checkpoint(filename='checkpoint.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return []

def process_dataset(dataset, azure_service, gcp_service, info_types):
    total_rows = len(dataset)
    progress = [0, 0.0, 0]

    processed_indices = load_checkpoint()
    start_index = len(processed_indices)

    def process_row(row, idx):
        if idx in processed_indices:
            return row

        start_time = time.time()

        try:
            text = row['generated_text']
            azure_result = azure_service.recognize_pii([text])
            gcp_result = gcp_service.recognize_pii([text], info_types)

            row['azure_results'] = azure_result
            row['gcp_results'] = gcp_result
        except Exception as e:
            progress[2] += 1
            print(f"Error processing row {idx}: {str(e)}")
            row['azure_results'] = None
            row['gcp_results'] = None

        end_time = time.time()
        row_time = end_time - start_time
        progress[0] += 1
        progress[1] += row_time

        pbar.update(1)
        pbar.set_postfix({
            'Avg Time/Row': f'{progress[1]/progress[0]:.2f}s',
            'Complete': f'{progress[0]/total_rows:.2%}',
            'Errors': progress[2]
        })

        return row

    print(f"Resuming from row {start_index}")
    pbar = tqdm(total=total_rows, initial=start_index, desc="Processing Rows", unit="row")

    processed_ds = dataset.map(process_row, with_indices=True)

    pbar.close()
    save_checkpoint(processed_indices)

    print(f"\nTotal processing time: {total_time:.2f}s")
    print(f"Average time per row: {total_time/total_rows:.2f}s")
    print(f"Total errors encountered: {progress[2]}")
    print(f"Processed {len(processed_indices)} out of {total_rows} rows")

    return processed_ds

def verify_huggingface_access(username):
    print("Verifying huggingface access..")
    api = HfApi()

    try:
        user_info = api.whoami()
        if user_info is None:
            raise ValueError("Not logging in to huggingface. Please run `huggingface-cli login`")
        if user_info['name'] != username:
            raise ValueError(f"Logged in username ({user_info['name']}) does not match provided username")
        print("Huggingface access verified successfully")
    except Exception as e:
        raise ValueError(f"Error verifying huggingface access: {str(e)}")


if __name__ == "__main__":
    hf_username = os.getenv("HUGGINGFACE_USERNAME")
    if not hf_username:
        raise ValueError("HUGGINGFACE_USERNAME not set in env variables")

    verify_huggingface_access(hf_username)

    # Load the dataset
    print("Loading dataset..")
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

    max_rows = 2500
    if len(filtered_ds) > max_rows:
        print(f"Filtered dataset too large. Max rows: {max_rows}, Current rows: {len(filtered_ds)}")
        sys.exit(1)

    azure_service = AzurePIIService()
    gcp_service = GCPPIIService()

    gcp_info_types = ["PERSON_NAME", "EMAIL_ADDRESS", "DATE", "STREET_ADDRESS", "ORGANIZATION_NAME"]

    processed_ds = process_dataset(filtered_ds, azure_service, gcp_service, gcp_info_types)

    processed_ds.save_to_disk("processed_dataset")

    repo_name = f"{hf_username}/processed-pii-finance-dataset"
    print(f"Uploading dataset to {repo_name}..")
    processed_ds.push_to_hub(repo_name)

    print(f"Dataset uploaded successfully to https://huggingface.co/datasets/{repo_name}")
    print(f"Processed {len(processed_ds)} rows")

