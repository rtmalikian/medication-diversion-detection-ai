# EHR Integration Guide — v2 status (2026-10-28, open.epic MCP added 2026-10-30)

## open.epic MCP — Epic documentation lookup (NOT patient data)

Epic publishes a remote Model Context Protocol (MCP) server at
**https://mcp.open.epic.com** (Streamable HTTP transport). v2 integrates it
in `v2/epic_mcp.py`, exposed as `GET /epic-mcp/tools` and
`POST /epic-mcp/docs`, to retrieve Epic's own API specifications and
integration guidance while building and running the FHIR integration below
— for example the Epic API specs for the MedicationRequest,
MedicationDispense and Observation APIs that `v2/fhir_client.py` consumes.

**What it is / is not.** The server returns *documentation content only*:
API specifications, interface summaries, additional documentation
(data-sharing playbooks, tutorials) and FAQs, including Epic on FHIR
content. Its tools (discovered at runtime via `tools/list`; live-verified
2026-10-30 against server "open.epic MCP" v4.0.9) are `search_documents`
and `get_document`. It is **not** a patient-data or FHIR-resource server:
no Patient/Observation/Encounter querying, no SMART/OAuth, no
credentials. **Patient data flows only through the FHIR R4 client +
SMART on FHIR in this repo — never through this MCP server.**

**No-PHI rule (enforced in code).** Epic states the server is *not
designated to handle protected health information (PHI) or other
confidential information* and that the user is responsible for anything
transmitted to it. `v2/epic_mcp.py` therefore screens every query before
sending: anything that looks like PHI / patient data (FHIR resource JSON,
`Patient/<id>` references, MRN/SSN-like patterns, date-of-birth phrasing)
is refused and the refusal logged. Only documentation-style queries (API
names, resource *type* names, spec topics) are allowed. Use is subject to
the open.epic Terms of Use.

**Config.** `EPIC_MCP_URL` (default `https://mcp.open.epic.com`),
`EPIC_MCP_ENABLED` (default on; set `0` to disable). The integration
degrades gracefully — if the server is unreachable or disabled, the rest
of the app (scoring, FHIR client, scribe) works unchanged.

**AI-client setup.** `epic-mcp.mcp.json.example` at the repo root is an
example MCP client config to copy into the file your client reads —
Claude Code reads `.mcp.json`, VS Code reads `.vscode/mcp.json`, and
Cursor reads its MCP config — it is an example, not a forced config.

Source: https://fhir.epic.com/documentation?docid=mcp



> **v2 implements the FHIR path described below.** `v2/fhir_client.py` is a working
> FHIR R4 client: it searches Patient, MedicationRequest, MedicationDispense,
> Encounter, Observation and Condition by patient, maps bundles to the v2 Chart
> model (`bundle_to_chart`), and the deterministic Layer A rules score from that
> data. It is tested against a local mock FHIR server (httpx MockTransport) and a
> bundled Synthea-format synthetic bundle (`tests_v2/fixtures/synthea_bundle.json`).
> API: `POST /fhir/score` (base URL + patient id), `POST /fhir/bundle/score`.
> Known limits: drug classification uses a name lexicon (production must bind
> RxNorm/ATC); SMART on FHIR OAuth is deployment configuration, not implemented
> here; HL7 v2 and vendor-specific (Epic/Cerner) adapters below remain plans.

---

# EHR Integration Guide for Clinical Risk Modeling Engine

This document outlines the approach for integrating the Clinical Risk Modeling Engine with major EHR systems like EPIC, Cerner, Allscripts, etc.

## Integration Approaches

### 1. FHIR (Fast Healthcare Interoperability Resources)
- **Standard API**: Use FHIR R4 or later for standardized healthcare data exchange
- **Patient Data**: Access patient demographics, medications, allergies, problems
- **Observations**: Retrieve lab results, vital signs, and other clinical observations
- **Security**: OAuth 2.0 authentication and SMART on FHIR for secure access

### 2. HL7 v2.x Integration
- **ADT Messages**: Patient admission, discharge, transfer messages
- **ORM/ORU**: Order entry and result reporting messages
- **Pharmacy-specific messages**: For medication-related data

### 3. Direct Database Integration
- **Read-only access**: Query EHR database tables directly (if allowed)
- **Scheduled sync**: Periodic data extraction to avoid performance impact
- **Real-time APIs**: For critical, real-time risk assessment

## EHR-Specific Considerations

### EPIC Integration
- **Epic Hyperspace**: Use Epic's native APIs
- **Epic App Orchard**: Consider the App Orchard for certified app integration
- **Clinical Decision Support (CDS) Hooks**: Implement CDS Hooks for real-time alerts
- **MyChart Integration**: Patient-facing risk notifications

### Cerner Integration
- **Cerner Millennium**: Use Cerner's Millennium APIs
- **PowerChart**: Integration with clinician-facing applications
- **HealtheIntent**: Leverage population health platform
- **Cerner Open Developer Experience (CODE)**: Use CODE APIs for custom integration

### Allscripts Integration
- **Enterprise EHR**: Use available APIs for data exchange
- **Prognos**: Consider for clinical analytics integration
- **TouchWorks**: Integration with practice management system

## Implementation Roadmap

### Phase 1: Data Access
1. Implement FHIR client for standardized data access
2. Create EHR-specific data adapters
3. Ensure compliance with data access policies

### Phase 2: Risk Calculation
1. Real-time risk assessment triggers
2. Batch processing for patient panels
3. Alert generation and workflow integration

### Phase 3: Clinical Workflow
1. Integration with clinician alerts
2. Patient dashboard integration
3. Reporting and analytics

## Security & Compliance

### HIPAA Compliance
- Data encryption in transit and at rest
- De-identification of clinical data
- Access controls and audit logs
- Business Associate Agreements (BAAs)

### Regulatory Requirements
- FDA software as medical device considerations
- Clinical decision support validation
- Evidence-based algorithm documentation
- Explainable AI requirements

## Technical Architecture

### API Gateway
- Standardized endpoints for EHR communication
- Rate limiting and throttling
- Request/response logging
- Error handling and retry logic

### Data Pipeline
- ETL processes for EHR data
- Data quality checks
- Standardization to common data models
- Risk score calculation

### Caching Layer
- Patient data caching to reduce EHR load
- Risk score caching for frequently accessed patients
- Cache invalidation strategies

## Implementation Considerations

### Performance
- Minimize EHR system impact
- Optimize query patterns
- Implement efficient data batching
- Consider edge computing for real-time scoring

### Reliability
- Fallback mechanisms when EHR is unavailable
- Data consistency across systems
- Error recovery procedures
- Monitoring and alerting

### Scalability
- Support for multiple EHR systems
- Multi-tenant architecture
- Horizontal scaling capabilities
- Geographic distribution if needed

## PDMP and CURES Integration

### Prescription Drug Monitoring Program (PDMP) Integration
- **State-Level Access**: Each state maintains its own PDMP with specific API requirements
- **Real-Time Monitoring**: Access to prescription history for controlled substances
- **Doctor Shopping Detection**: Cross-reference prescriptions from multiple prescribers
- **High-Risk Pattern Recognition**: Identify frequent, high-dose, or overlapping prescriptions

### CURES Integration (California's PDMP)
- **API Integration**: CURES offers API access for automated data retrieval
- **Daily Uploads**: Healthcare providers must upload prescription data daily
- **Prescriber Access**: Authorized prescribers can access patient prescription history
- **Risk Factors**: Monitor for:
  - Multiple prescribers for controlled substances
  - High-dose opioid prescriptions (MME calculations)
  - Concurrent benzodiazepine and opioid prescribing
  - Short-interval refills and early requests

### PDMP/CURES Data Elements for Risk Assessment

#### Doctor Shopping Indicators
- Number of different prescribers per time period
- Number of different pharmacies used
- Geographic distribution of prescriptions
- Time intervals between prescriptions from different providers

#### High-Risk Medication Patterns
- Total morphine milligram equivalents (MME) per day
- Concurrent use of multiple controlled substances
- High-dose prescribing patterns (≥90 MME/day for opioids)
- Frequency of Schedule II controlled substance prescriptions

#### Red Flag Behaviors
- Early refill requests across multiple pharmacies
- Prescription quantities inconsistent with treatment plan
- Out-of-state prescriptions when local providers available
- Patterns of lost/stolen prescription reports

### Technical Integration Approach

#### API-Based Integration
```python
class PDMPIntegration:
    def __init__(self, state, credentials):
        self.state = state
        self.credentials = credentials

    def get_prescription_history(self, patient_info):
        # Query PDMP/CURES for prescription history
        pass

    def analyze_patterns(self, prescription_data):
        # Detect doctor shopping, high-dose patterns, etc.
        pass

    def flag_high_risk_behaviors(self, patient_id):
        # Generate risk flags based on PDMP data
        pass
```

#### Batch Processing vs Real-Time
- **Real-Time**: Query PDMP at time of prescription for immediate risk assessment
- **Batch Processing**: Periodic analysis of all patients for pattern recognition
- **Scheduled Monitoring**: Regular updates of risk scores based on new PDMP data

### Regulatory and Privacy Considerations
- **HIPAA Compliance**: PDMP/CURES data is subject to special privacy protections
- **Consent Requirements**: Some states require patient consent for PDMP queries
- **Permissible Use**: Only authorized healthcare providers can access PDMP data
- **Audit Requirements**: Maintain logs of all PDMP queries for compliance

### Implementation Benefits
- Enhanced detection of medication diversion
- Comprehensive patient medication history
- Evidence-based risk scoring with external validation
- Compliance with state monitoring requirements

This integration enhances the Clinical Risk Modeling Engine by providing access to external prescription monitoring data that is crucial for identifying medication diversion patterns and high-risk behaviors that may not be visible within a single healthcare system.

## Testing Strategy

### Integration Testing
- Mock EHR APIs for development
- Staging environment with actual EHR data
- Security penetration testing
- Performance and load testing

### Clinical Validation
- Retrospective validation studies
- Prospective pilot programs
- Clinician feedback integration
- Algorithm bias detection

## Deployment Considerations

### On-Premise vs Cloud
- On-premise for security-sensitive environments
- Cloud for scalability and maintenance
- Hybrid approach for flexibility

### Maintenance
- Regular model retraining
- EHR API version updates
- Clinical guideline updates
- Performance monitoring

This integration would allow the Clinical Risk Modeling Engine to function as a comprehensive solution within existing healthcare IT infrastructures, supporting the identification of medication diversion risks across different EHR platforms.