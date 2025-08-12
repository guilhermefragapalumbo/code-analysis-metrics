# Towards Measurable Ethics: A Framework for AI Ethical Compliance Metrics
### by Guilherme Palumbo

The methodology consists of five sequential stages: (1) AI Lifecycle Stage Identification, (2) Tasks Definition, (3) Compliance List Development, (4) Compliance Checks Distribution, and (5) Metrics Creation. 
Each stage builds upon the previous one to form a comprehensive architecture capable of producing objective, interpretable, and quantifiable metrics grounded in ethical principles, grounded in both AI development best practices and the Ethics Guidelines for Trustworthy AI.



<div style="display: flex; gap: 15px; justify-content: center; margin: 20px 0;">
  <a href="https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng" 
     style="
       background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
       color: white;
       padding: 12px 30px;
       border-radius: 30px;
       text-align: center;
       text-decoration: none;
       font-weight: 600;
       font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
       box-shadow: 0 4px 12px rgba(37, 117, 252, 0.4);
       transition: background 0.3s ease, box-shadow 0.3s ease;
     "
     onmouseover="this.style.background='linear-gradient(135deg, #4e07b7 0%, #1a5edd 100%)'; this.style.boxShadow='0 6px 20px rgba(30, 94, 221, 0.6)';"
     onmouseout="this.style.background='linear-gradient(135deg, #6a11cb 0%, #2575fc 100%)'; this.style.boxShadow='0 4px 12px rgba(37, 117, 252, 0.4)';"
  >
    AI Act
  </a>

  <a href="https://digital-strategy.ec.europa.eu/en/library/ethics-guidelines-trustworthy-ai" 
     style="
       background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
       color: white;
       padding: 12px 30px;
       border-radius: 30px;
       text-align: center;
       text-decoration: none;
       font-weight: 600;
       font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
       box-shadow: 0 4px 12px rgba(255, 75, 43, 0.4);
       transition: background 0.3s ease, box-shadow 0.3s ease;
     "
     onmouseover="this.style.background='linear-gradient(135deg, #d83655 0%, #d63b2e 100%)'; this.style.boxShadow='0 6px 20px rgba(214, 59, 46, 0.6)';"
     onmouseout="this.style.background='linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%)'; this.style.boxShadow='0 4px 12px rgba(255, 75, 43, 0.4)';"
  >
    Ethics Guideline for Trustworthy AI
  </a>
</div>


## (1) AI Lifecycle Stage Identification



![Alt text](images/circular_pipeline.jpg)

ou

![Alt text](images/pipeline.jpg)

## (2) Tasks Definition

| Task Name | Description | Stage |
|-----------|-------------|-------|
| Data Collection | Gather raw data from databases, APIs, sensors, web scraping, user inputs, or other sources. | Data Ingestion & Versioning |
| Data Extraction | Retrieve data from structured (CSV, SQL) and unstructured (PDFs, images, text) sources. | Data Ingestion & Versioning |
| Data Integration | Merge data from multiple sources while resolving inconsistencies. | Data Ingestion & Versioning |
| Streaming Data Handling | Manage real-time ingestion from IoT devices, event-driven systems, or streaming platforms. | Data Ingestion & Versioning |
| Batch Data Handling | Process large amounts of data in scheduled or on-demand batches. | Data Ingestion & Versioning |

## (3) Compliance List Development

| Task Name | Compliance Check | Compliance Description |
|-----------|------------------|------------------------|
| Data Collection | Raw Data Acquisition | Check if the code includes methods to collect raw data from databases, APIs, sensors, web scraping, or user inputs. |
| Data Extraction | Data Retrieval Implementation | Verify if the code includes logic to extract data from structured (CSV, SQL databases) and unstructured (PDFs, images, text) sources. |
| Data Integration | Multi-Source Data Merging | Assess whether the code includes functionality to merge multiple data sources while resolving inconsistencies. |
| Streaming Data Handling | Real-Time Data Ingestion | Determine if the code implements handling of real-time data streams from IoT devices, event-driven systems, or streaming platforms. |
| Batch Data Handling | Batch Processing Implementation | Check if the code supports batch processing mechanisms for large-scale data ingestion. |

## (4) Compliance Checks Distribution

