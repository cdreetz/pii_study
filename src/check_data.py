from datasets import load_dataset

# Load the dataset
ds = load_dataset("gretelai/synthetic_pii_finance_multilingual")

# Define the filter conditions
document_types = ["Email", "IT support ticket", "Customer support conversational log"]
pii_types = ["company", "email", "date", "street_address", "name"]
language = "English"

# Apply the filters
filtered_ds = ds.filter(
    lambda example: (
        example['document_type'] in document_types and
        example['language'] == language
    )
)

# Print some information about the filtered dataset
print(f"Original dataset size: {len(ds['train'])}")
print(f"Filtered dataset size: {len(filtered_ds['train'])}")

# Display a few examples from the filtered dataset
print("\nSample entries from the filtered dataset:")
for i, example in enumerate(filtered_ds['train'].select(range(5))):
    print(f"\nExample {i + 1}:")
    print(f"Document Type: {example['document_type']}")
    print(f"Language: {example['language']}")
    #print(f"Text: {example['generated_text'][:100]}...")  # Display first 100 characters of the text
