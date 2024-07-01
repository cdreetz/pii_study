#!/usr/bin/python3
import argparse
from redact import redact_text

item = "Hey Alex, what time will you be arriving at SFO?"

response = redact_text(item, region_name='us-east-1')
print(response)
