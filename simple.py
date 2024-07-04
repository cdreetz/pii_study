from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import google.cloud.dlp
from dotenv import load_dotenv
from typing import List
import sys
import os

load_dotenv()


class GCPPIIService:
    def __init__(self):
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.client = google.cloud.dlp_v2.DlpServiceClient()
        self.parent = f"projects/{self.project_id}/locations/global"

    def recognize_pii(self, documents, info_types):
        inspect_config = {
            "info_types": [{"name": info_type} for info_type in info_types]
        }

        results = []
        for document in documents:
            response = self.client.inspect_content(
                    request = {
                        "parent": self.parent,
                        "inspect_config": inspect_config,
                        "item": {"value": document},
                    }
            )
            results.append(response)
        return results

    def print_pii_results(self, results):
        for i, response in enumerate(results):
            print(f"Document {i + 1}:")
            if response.result.findings:
                for finding in response.result.findings:
                    print(f"Entity: {finding.quote}")
                    print(f"    Category: {finding.info_type.name}")
                    print(f"    Confidence Score: {finding.likelihood}")
                    print(f"    Location: {finding.location.byte_range}")
            else:
                print("No findings")
            print("---")

    def process_documents(self, documents, info_types):
        results = self.recognize_pii(documents, info_types)
        self.print_pii_results(results)




class AzurePIIService:
    def __init__(self):
        self.client = TextAnalyticsClient(
                endpoint=os.getenv("AZURE_ENDPOINT"), 
                credential=AzureKeyCredential(os.getenv("AZURE_API_KEY"))
        )

    def recognize_pii(self, documents, language="en"):
        response = self.client.recognize_pii_entities(documents, language=language)
        return [doc for doc in response if not doc.is_error]

    def print_pii_results(self, results):
        for doc in results:
            print(doc)
            print(f"Redacted text: {doc.redacted_text}")
            for entity in doc.entities:
                print(f"Entity: {entity.text}")
                print(f"    Category: {entity.category}")
                print(f"    Confidence Score: {entity.confidence_score}")
                print(f"    Offset: {entity.offset}")
                print(f"    Length: {entity.length}")

    def process_documents(self, documents):
        results = self.recognize_pii(documents)
        self.print_pii_results(results)


def process_documents_with_service(service_name, documents, info_types=None):
    if service_name.lower() == 'azure':
        service = AzurePIIService()
        service.process_documents(documents)
    elif service_name.lower() == 'gcp':
        if info_types is None:
            raise ValueError("Info types must be provided for GCP DLP service")
        service = GCPPIIService()
        service.process_documents(documents, info_types)
    else:
        raise ValueError("Invalid service name. Choose 'azure' or 'gcp'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <service>")
        print(" <service>: 'azure' or 'gcp'")
        sys.exit(1)

    service = sys.argv[1].lower()

    documents = [
        "Hey Alex, what time will you be arriving at SFO?",
        "The employee's SSN is 859-98-0987 and their last name is Jones.",
        "The employee's phone number is 555-555-5555 and their email is rob.jones@gmail.com."
    ]

    info_types = [
        "PERSON_NAME", 
        "LOCATION", 
        "EMAIL_ADDRESS", 
        "US_INDIVIDUAL_TAXPAYER_IDENTIFICATION_NUMBER"
    ]

    if service == 'azure':
        process_documents_with_service('azure', documents)
    elif service == 'gcp':
        process_documents_with_service('gcp', documents, info_types)
    else:
        print(f"Error: Invalid service '{service}'. Choose 'azure' or 'gcp'.")
        sys.exit(1)


# :!env/bin/python %
