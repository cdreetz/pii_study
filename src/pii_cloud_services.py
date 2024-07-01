import abc
from typing import List, Dict
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class PIICloudService(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def extract_pii(self, text: str) -> List[Dict[str, str]]:
        """Extract PII from the given text."""
        pass

    @abc.abstractmethod
    def redact_pii(self, text: str) -> str:
        """Redact PII from the given text."""
        pass

class AzurePIIService(PIICloudService):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("AZURE_API_KEY")
        self.endpoint = os.getenv("AZURE_ENDPOINT")
        if not self.api_key or not self.endpoint:
            raise ValueError("AZURE_API_KEY and AZURE_ENDPOINT must be set in .env file")

    def extract_pii(self, text: str) -> List[Dict[str, str]]:
        # Import the required Azure libraries
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.core.credentials import AzureKeyCredential

        # Create a Text Analytics client
        client = TextAnalyticsClient(
            credential=AzureKeyCredential(self.api_key),
            endpoint=self.endpoint
        )

        # Call the recognize_pii_entities method
        result = client.recognize_pii_entities([text], language="en")[0]

        # Format the results
        pii_entities = [
            {
                "entity": entity.text,
                "category": entity.category,
                "confidence_score": entity.confidence_score,
                "start": entity.offset,
                "end": entity.offset + len(entity.text)
            } for entity in result.entities
        ]

        return pii_entities

    def redact_pii(self, text: str) -> str:
        # Import the required Azure libraries
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.core.credentials import AzureKeyCredential

        # Create a Text Analytics client
        client = TextAnalyticsClient(
            credential=AzureKeyCredential(self.api_key),
            endpoint=self.endpoint
        )

        # Call the recognize_pii_entities method with redaction enabled
        result = client.recognize_pii_entities([text], language="en")[0]

        # Return the redacted text
        return result.redacted_text

class GCPPIIService(PIICloudService):
    def __init__(self):
        super().__init__()
        self.project_id = os.getenv("GCP_PROJECT_ID")
        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID must be set in .env file")

    def extract_pii(self, text: str) -> List[Dict[str, str]]:
        # Import the required GCP libraries
        from google.cloud import dlp_v2

        # Create DLP client
        client = dlp_v2.DlpServiceClient()
        parent = f"projects/{self.project_id}/locations/global"
        info_types = [
            "PERSON_NAME", 
            "LOCATION", 
            "EMAIL_ADDRESS", 
            "US_INDIVIDUAL_TAXPAYER_IDENTIFICATION_NUMBER"
        ]
        info_types = [{"name": info_type} for info_type in info_types]
        inspect_config = {
            "info_types": info_types,
            "include_quote": True
        }

        # Call the inspect_content method
        response = client.inspect_content(
            request={
                "parent": parent, 
                "item": {"value": text}, 
                "inspect_config": inspect_config
            }
        )

        # Format the results
        pii_entities = [
            {
                "entity": finding.quote,
                "category": finding.info_type.name,
                "likelihood": finding.likelihood.name,
                "start": finding.location.byte_range.start,
                "end": finding.location.byte_range.end
            }
            for finding in response.result.findings
        ]

        return pii_entities

    def redact_pii(self, text: str) -> str:
        # Import the required GCP libraries
        from google.cloud import dlp_v2

        # Create DLP client
        client = dlp_v2.DlpServiceClient()
        parent = f"projects/{self.project_id}/locations/global"
        item = {"value": text}
        deidentify_config = {
            "info_type_transformations": {
                "transformations": [
                    {"primitive_transformation": {
                        "replace_config": {
                            "new_value": {
                                "string_value": "[REDACTED]"
                            }
                        }
                    }}
                ]
            }
        }

        # Call the deidentify_content method
        response = client.deidentify_content(
            request={
                "parent": parent, 
                "item": item, 
                "deidentify_config": deidentify_config
            }
        )

        # Return the redacted text
        return response.item.value

# Example usage
if __name__ == "__main__":
    azure_service = AzurePIIService()
    gcp_service = GCPPIIService()

    sample_text = "John Doe's SSN is 123-45-6789 and his email is johndoe@example.com"

    # Azure PII extraction and redaction
    azure_extracted_pii = azure_service.extract_pii(sample_text)
    azure_redacted_text = azure_service.redact_pii(sample_text)

    # GCP PII extraction and redaction
    gcp_extracted_pii = gcp_service.extract_pii(sample_text)
    gcp_redacted_text = gcp_service.redact_pii(sample_text)

    print("Azure Extracted PII:", azure_extracted_pii)
    print("Azure Redacted Text:", azure_redacted_text)
    print("GCP Extracted PII:", gcp_extracted_pii)
    print("GCP Redacted Text:", gcp_redacted_text)