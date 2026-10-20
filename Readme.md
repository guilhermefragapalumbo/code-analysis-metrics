# Framework for AI Ethical Compliance Metrics

Code and data for the article *A Lifecycle Framework for Automated Assessment of Trustworthy AI Requirements in Machine Learning Codebases*.

Guilherme Palumbo¹ ([ORCID](https://orcid.org/0000-0001-6633-3033)), Davide Carneiro² ([ORCID](https://orcid.org/0000-0002-6650-0388)), Victor Alves³ ([ORCID](https://orcid.org/0000-0003-1819-7051))

¹ CIICESI, ESTG, Instituto Politécnico do Porto, Felgueiras, Portugal\
² INESC TEC: Instituto de Engenharia de Sistemas e Computadores, Tecnologia e Ciência, Porto, Portugal\
³ ALGORITMI Research Centre / LASI, University of Minho, Braga, Portugal

Corresponding author: Guilherme Palumbo (gfp@estg.ipp.pt)

The methodology consists of five sequential stages: (1) AI Lifecycle Stage Identification, (2) Tasks Definition, (3) Compliance List Development, (4) Compliance Checks Distribution, and (5) Metrics Creation.
Each stage builds upon the previous one to form a comprehensive architecture capable of producing objective, interpretable, and quantifiable metrics based on ethical principles, focused on AI development best practices, the [EU AI Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng) and the [Ethics Guidelines for Trustworthy AI](https://digital-strategy.ec.europa.eu/en/library/ethics-guidelines-trustworthy-ai).

## Repository contents

| Folder                                                                          | Content                                                                                                          |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| [Section (1) below](#1-ai-lifecycle-stage-identification)                        | (1) The seven lifecycle stages                                                                                   |
| [`(2) tasks_definition`](<./(2)%20tasks_definition>)                           | (2) The 69 tasks                                                                                                 |
| [`(3) compliance_check_development`](<./(3)%20compliance_check_development>)   | (3) The 69 compliance checks                                                                                     |
| [`(4) compliance_check_distribution`](<./(4)%20compliance_check_distribution>) | (4) Mapping of compliance checks to ethical elements, with the reasoning for each association                    |
| [`(5) metrics`](<./(5)%20metrics>)                                             | (5) The 41 metrics: definitions and implementation                                                               |
| [`(6) preliminary_validation`](<./(6)%20preliminary_validation>)               | Evaluation script and prompts, all LLM outputs, and a script that recomputes the results tables                  |
| [`ML Projects`](<./ML%20Projects>)                                             | The two projects evaluated in the preliminary validation, before and after the interventions                     |
| [`(0) ethical_principles`](<./(0)%20ethical_principles>)                       | The requirements and elements of the Ethics Guidelines for Trustworthy AI, with their definitions, used as input |

**Reproducing the results.** With Python 3.8 or later and no extra packages:

```bash
cd "(6) preliminary_validation/analysis"
python reproduce.py
```

This recomputes the article's Tables 3–6 from the archived LLM outputs and writes a cell-by-cell comparison to
`output/REPRODUCIBILITY_REPORT.md`. The evaluation protocol (model, parameters, prompts, parsing) is documented in
[`(6) preliminary_validation/README.md`](<./(6)%20preliminary_validation/README.md>).

## (1) AI Lifecycle Stage Identification

Seven stages were identified that capture the end-to-end lifecycle of AI system development (represented in the Figure below):

| Stage                                   | Description                                                                                                                                                             |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data Ingestion and Versioning** | Collection and management of raw data, integrating it from various systems, and applying appropriate version control ensuring data traceability and reproducibility.    |
| **Data Preprocessing**            | Cleaning, transforming, and preparing data for training. Tasks may include outlier removal, feature encoding, and handling missing values.                              |
| **Model Training and Tuning**     | Learning algorithms are applied to data, and models are developed and trained. Tasks include hyperparameter optimization, regularization, and validation splits.        |
| **Model Evaluation**              | Validating model performance using appropriate metrics.                                                                                                                 |
| **Model Versioning**              | As models are updated and retrained, versioning ensures changes are tracked and documented, managing different model versions.                                          |
| **Model Deployment**              | Integrating the model into production environments. This may involve wrapping models in APIs, ensuring infrastructure compatibility, and conducting deployment testing. |
| **Monitoring and Maintenance**    | Ongoing observation and updates to the model post-deployment.                                                                                                           |

![Alt text](images/pipeline.jpg)

## (2) Tasks Definition

| Task Name               | Description                                                                                 | Stage                         |
| ----------------------- | ------------------------------------------------------------------------------------------- | ----------------------------- |
| Data Collection         | Gather raw data from databases, APIs, sensors, web scraping, user inputs, or other sources. | Data Ingestion and Versioning |
| Data Extraction         | Retrieve data from structured (CSV, SQL) and unstructured (PDFs, images, text) sources.     | Data Ingestion and Versioning |
| Data Integration        | Merge data from multiple sources while resolving inconsistencies.                           | Data Ingestion and Versioning |
| Streaming Data Handling | Manage real-time ingestion from IoT devices, event-driven systems, or streaming platforms.  | Data Ingestion and Versioning |
| Batch Data Handling     | Process large amounts of data in scheduled or on-demand batches.                            | Data Ingestion and Versioning |

*Table continues... See the full table [here](<./(2)%20tasks_definition/tasks_definition.md>)*

Tasks per stage:

| Stage                         | Tasks        |
| ----------------------------- | ------------ |
| Data Ingestion and Versioning | 10           |
| Data Preprocessing            | 14           |
| Model Training and Tuning     | 11           |
| Model Evaluation              | 11           |
| Model Versioning              | 6            |
| Model Deployment              | 9            |
| Monitoring and Maintenance    | 8            |
| **Total**               | **69** |

## (3) Compliance List Development

| Task Name               | Compliance Check                | Compliance Description                                                                                                                |
| ----------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Data Collection         | Raw Data Acquisition            | Check if the code includes methods to collect raw data from databases, APIs, sensors, web scraping, or user inputs.                   |
| Data Extraction         | Data Retrieval Implementation   | Verify if the code includes logic to extract data from structured (CSV, SQL databases) and unstructured (PDFs, images, text) sources. |
| Data Integration        | Multi-Source Data Merging       | Assess whether the code includes functionality to merge multiple data sources while resolving inconsistencies.                        |
| Streaming Data Handling | Real-Time Data Ingestion        | Determine if the code implements handling of real-time data streams from IoT devices, event-driven systems, or streaming platforms.   |
| Batch Data Handling     | Batch Processing Implementation | Check if the code supports batch processing mechanisms for large-scale data ingestion.                                                |

*Table continues... See the full table [here](<./(3)%20compliance_check_development/compliance_list.md>)*

## (4) Compliance Checks Distribution

Example: Accountability Auditability

| Task Name                      | Compliance Check                 | Reasoning                                                                                                                                                                                                                                                                                                  |
| ------------------------------ | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Metadata Management            | Metadata Storage and Tracking    | Metadata management directly supports auditability by providing a clear, structured record of the data used in an AI system. This allows auditors to understand the sources, types, and characteristics of the data, which is crucial for assessing the fairness, accuracy, and reliability of the system. |
| Data Versioning                | Version Control Implementation   | Data versioning is essential for auditability as it allows for the tracking of changes made to the data over time. This is crucial for understanding how the AI system's performance and behavior may have been influenced by changes in the data.                                                         |
| Schema Validation              | Schema Conformance Check         | Schema validation ensures that the data adheres to a predefined schema, which is important for auditability. This allows auditors to understand the structure and format of the data, and to check that the data has been processed and used correctly.                                                    |
| Data Quality Checks            | Consistency and Integrity Checks | Data quality checks are key to auditability as they ensure the integrity and consistency of the data. This allows auditors to assess the quality of the data and to understand any potential issues or biases that may affect the AI system's performance.                                                 |
| Performance Metric Calculation | Evaluation Metric Computation    | The calculation of performance metrics allows auditors to assess the effectiveness and accuracy of the AI system. This is crucial for auditability as it provides a quantitative measure of the system's performance.                                                                                      |

*Table continues... See the all the reasoning [here](<./(4)%20compliance_check_distribution/compliance_check_distribution_reasoning_markdown>)*

| Task Name                      | Compliance Check                 | Compliance Description                                                                                                                     |
| ------------------------------ | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Metadata Management            | Metadata Storage and Tracking    | Verify if the code maintains structured metadata, including source details, timestamps, and schema definitions.                            |
| Data Versioning                | Version Control Implementation   | Ensure that the code implements dataset versioning to track different versions for reproducibility and rollback capabilities.              |
| Schema Validation              | Schema Conformance Check         | Check if the code validates incoming data against predefined formats and structures to ensure compliance.                                  |
| Data Quality Checks            | Consistency and Integrity Checks | Assess whether the code includes checks to maintain data quality and consistency throughout preprocessing.                                 |
| Performance Metric Calculation | Evaluation Metric Computation    | Check if the code computes relevant metrics such as accuracy, precision, recall, F1-score for classification, or RMSE, MAE for regression. |

*Table continues... See the all the tables [here](<./(4)%20compliance_check_distribution/markdown>)*

Coverage: this version of the mapping covers 21 of the 23 ethical elements. *Accessibility and universal design* and
*Redress* are not included in this version; they are covered in the newer versions of the mapping.

## (5) Metrics Creation

Metrics aggregate the 0–5 compliance-check scores according to a defined rule (mean, weighted mean, minimum, median,
proportion above a threshold, dispersion and others). 41 metrics are defined and implemented in code; the preliminary
validation reports 23 of them (Tables 4 and 6).

Example:

| Metric               | Formula                                     | Relevant tasks                         |
| -------------------- | ------------------------------------------- | -------------------------------------- |
| Data Integrity Score | min(Data Quality Checks, Schema Validation) | Data Quality Checks, Schema Validation |

*See all 41 metrics [here](<./(5)%20metrics/metrics.md>) and the machine-readable definitions in [`(5) metrics`](<./(5)%20metrics>).*

## (6) Preliminary Validation

Two student ML projects (*Food Allergy Analysis* and *House Price Forecast*) were evaluated before and after code changes
guided by the LLM's recommendations. The folder contains the evaluation script and prompts exactly as used, every LLM
output, the per-task scores, and `analysis/reproduce.py`, which recomputes the article's tables. Known limitations of the
archived data are listed in [`(6) preliminary_validation/README.md`](<./(6)%20preliminary_validation/README.md#5-known-limitations>).

## License

The material in this repository is licensed under [CC BY 4.0](./LICENSE), with two exceptions:

- The two student projects in [`ML Projects`](<./ML%20Projects>) remain the work of their authors and are included only so
  that the evaluation can be reproduced.
- The Food Allergy dataset is subject to the licence of its original Zenodo distribution.
