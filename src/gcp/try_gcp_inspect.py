from typing import List
from dotenv import load_dotenv
import google.cloud.dlp
import os

load_dotenv()
project = os.getenv("project")

def inspect_and_get_positions(
        project: str, item: str, info_types: List[str]
) -> None:
    """
    Uses the Data Loss Prevention API to inspect senstitive data in a
    string and retur the positions of detected info types.
    Args:
        project: The Google Cloud project idto use as a parent resource
        item: The string to inspect
        info_types: A list of string representing info types to look for
    Returns:
        None; the response from the API is printed to the terminal
    """
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    parent = f"projects/{project}/locations/global"
    info_types = [{"name": info_type} for info_type in info_types]
    inspect_config = {
            "info_types": info_types
    }

    response = dlp.inspect_content(
            request={
                "parent": parent,
                "inspect_config": inspect_config,
                "item": {"value": item},
            }
    )
    print(response)

    #if response.result.findings:
    #    print("Findings:")
    #    for finding in response.result.findings:
    #        start = finding.location.byte_range.start
    #        end = finding.location.byte_range.end
    #        print(f"Info type: {finding.info_type.name}")
    #        print(f"Likelihood: {finding.likelihood}")
    #        print(f"Quote: {item[start:end]}")
    #        print(f"Position: start={start}, end={end}")
    #        print()
    #else:
    #    print("No findings.")

item = "Hey Alex, what time will you be arriving at SFO?"
info_types = ["PERSON_NAME", "LOCATION", "EMAIL_ADDRESS", "US_INDIVIDUAL_TAXPAYER_IDENTIFICATION_NUMBER"]
inspect_and_get_positions(project, item, info_types)


