# How to use

Add your keys vars to a .env file, following the .env.example 

`src/main.py` will use both the Azure ang GCP services and iterate over the pulled HF dataset, and save the results in their corresponding columns

`src/check_data.py` can be used to examine the data pulled during the study

`src/pii_services.py` can be used to test each of the services, it should be run like `python src/pii_services.py azure`

Find the resulting dataset after processing [here](https://huggingface.co/datasets/cdreetz/filtered-pii-results)

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



## Results
```
Azure Results:
Precision: 0.8165
Recall: 0.7622
F1 Score: 0.7592

GCP Results:
Precision: 0.8635
Recall: 0.7755
F1 Score: 0.7974

PII Type Counts:

Email:
  True labels: 1130
  Azure detected: 721
  GCP detected: 1110

Name:
  True labels: 4124
  Azure detected: 4443
  GCP detected: 5215

Street_address:
  True labels: 1458
  Azure detected: 878
  GCP detected: 1013

Phone_number:
  True labels: 385
  Azure detected: 389
  GCP detected: 0

Date:
  True labels: 5390
  Azure detected: 4109
  GCP detected: 4529
```
