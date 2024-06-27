# PII Extraction and Redaction Study

With the increasing number of user facing conversational AI applications, 
there will be more text data to deal with than ever.  One option is to 
simply log or store the inputs and outputs of these applications, leaving 
product usage data to consider only the number of requests themselves, and 
not thier content. The other option being to store both requests and the 
contents, including both user inputs and AI generated responses.  While 
there is no concern for some to do this, there are still a lot of companies 
that deal with data privacy concerns, one of which being the storage of PII, 
or personally identifiable information.

Given how valuable this in-depth product usage data would be in analysis 
and attempts to improve the product, one may want to implement a solution that 
identifies and redacts PII in chat data. Luckily each of the main cloud 
providers, as well as a number of software companies provide services exactly 
for this, or developing your own is also an option. In this study we will 
evaluate these services from all providers as well as open source solutions. 
(Possibly attempting to train our own.)

## The data
To evaluate these solutions, we first need some data and a metric to measure.
We are starting with a large open source [dataset](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual?row=1) off Huggingface, provided by GretelAI.
From this dataset we filter it down for data more relevant to our case study,
with a focus on conversational type texts, in English, and with specific PII types
like name, email, location, and SSN or 'US Taxpayer ID'

The file `check_data.py` contains a script that pulls this data with the same filters used. 


## Azure

The file at `cloud/azure/try_azure.py` contains an example script of how you 
can get started using the PII extraction solution by Azure. All you need are 
the necessary servcies created in your Azure environment, and the variables 
listed under the Azure section of the `.env.example`


## GCP

Similar thing with the `cloud/gcp/try_gcp.py` except in this case you have to 
setup GCP auth on your local machine, then provide the project name your 
authorized account has access to. You might also need to enable the APIs.


## AWS