from datasets import load_dataset
import re
from collections import defaultdict
import json

# Define the PII types we're interested in
PII_TYPES = {'name', 'email', 'date', 'phone_number', 'street_address'}

# Mapping dictionaries
azure_mapping = {
    'Person': 'name',
    'PersonType': 'name',
    'Organization': 'name',
    'Address': 'street_address',
    'Email': 'email',
    'DateTime': 'date',
    'Date': 'date',
    'PhoneNumber': 'phone_number',
}

gcp_mapping = {
    'PERSON_NAME': 'name',
    'ORGANIZATION_NAME': 'name',
    'STREET_ADDRESS': 'street_address',
    'EMAIL_ADDRESS': 'email',
    'DATE': 'date',
    'PHONE_NUMBER': 'phone_number',
}

def map_entity_type(entity_type, mapping):
    return mapping.get(entity_type, entity_type.lower())

def extract_entity_types(result_string, mapping):
    pattern = r"category=(.*?),"
    entities = re.findall(pattern, result_string)
    return [map_entity_type(entity, mapping) for entity in entities if map_entity_type(entity, mapping) in PII_TYPES]

def extract_gcp_entity_types(result_string, mapping):
    pattern = r"name: \"(.*?)\""
    entities = re.findall(pattern, result_string)
    return [map_entity_type(entity, mapping) for entity in entities if map_entity_type(entity, mapping) in PII_TYPES]

def calculate_metrics(true_entities, predicted_entities):
    true_set = set(true_entities)
    pred_set = set(predicted_entities)
    true_positives = len(true_set & pred_set)
    false_positives = len(pred_set - true_set)
    false_negatives = len(true_set - pred_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def count_names(entities):
    name_count = 0
    for i, entity in enumerate(entities):
        if entity == 'name':
            if i == 0 or entities[i-1] != 'name':
                name_count += 1
    return name_count

def evaluate_services(dataset):
    azure_metrics = defaultdict(list)
    gcp_metrics = defaultdict(list)
    pii_counts = defaultdict(lambda: defaultdict(int))
    
    for i, sample in enumerate(dataset['train']):
        true_entities = [entity['label'] for entity in json.loads(sample['pii_spans']) if entity['label'] in PII_TYPES]
        azure_entities = extract_entity_types(sample['azure_results'], azure_mapping)
        gcp_entities = extract_gcp_entity_types(sample['gcp_results'], gcp_mapping)
        
        # Count names
        pii_counts['true']['name'] += count_names(true_entities)
        pii_counts['azure']['name'] += count_names(azure_entities)
        pii_counts['gcp']['name'] += count_names(gcp_entities)
        
        # Count other entity types
        for entity in true_entities:
            if entity != 'name':
                pii_counts['true'][entity] += 1
        for entity in azure_entities:
            if entity != 'name':
                pii_counts['azure'][entity] += 1
        for entity in gcp_entities:
            if entity != 'name':
                pii_counts['gcp'][entity] += 1
        
        azure_precision, azure_recall, azure_f1 = calculate_metrics(true_entities, azure_entities)
        gcp_precision, gcp_recall, gcp_f1 = calculate_metrics(true_entities, gcp_entities)
        
        azure_metrics['precision'].append(azure_precision)
        azure_metrics['recall'].append(azure_recall)
        azure_metrics['f1'].append(azure_f1)
        
        gcp_metrics['precision'].append(gcp_precision)
        gcp_metrics['recall'].append(gcp_recall)
        gcp_metrics['f1'].append(gcp_f1)
        
        if i < 5:  # Print details for the first 5 samples
            print(f"\nSample {i+1}:")
            print(f"True entities: {true_entities}")
            print(f"Azure entities: {azure_entities}")
            print(f"GCP entities: {gcp_entities}")
            print(f"True names: {count_names(true_entities)}")
            print(f"Azure names: {count_names(azure_entities)}")
            print(f"GCP names: {count_names(gcp_entities)}")
    
    return azure_metrics, gcp_metrics, pii_counts

def print_results(azure_metrics, gcp_metrics, pii_counts):
    print("\nAzure Results:")
    print(f"Precision: {sum(azure_metrics['precision']) / len(azure_metrics['precision']):.4f}")
    print(f"Recall: {sum(azure_metrics['recall']) / len(azure_metrics['recall']):.4f}")
    print(f"F1 Score: {sum(azure_metrics['f1']) / len(azure_metrics['f1']):.4f}")
    
    print("\nGCP Results:")
    print(f"Precision: {sum(gcp_metrics['precision']) / len(gcp_metrics['precision']):.4f}")
    print(f"Recall: {sum(gcp_metrics['recall']) / len(gcp_metrics['recall']):.4f}")
    print(f"F1 Score: {sum(gcp_metrics['f1']) / len(gcp_metrics['f1']):.4f}")
    
    print("\nPII Type Counts:")
    for pii_type in PII_TYPES:
        print(f"\n{pii_type.capitalize()}:")
        print(f"  True labels: {pii_counts['true'][pii_type]}")
        print(f"  Azure detected: {pii_counts['azure'][pii_type]}")
        print(f"  GCP detected: {pii_counts['gcp'][pii_type]}")

# Main execution
dataset = load_dataset("cdreetz/filtered-pii-results")
azure_metrics, gcp_metrics, pii_counts = evaluate_services(dataset)
print_results(azure_metrics, gcp_metrics, pii_counts)
