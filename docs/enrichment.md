# Enrichment

Enrichers transform `TraceEvent`s before they reach sinks. They run in sequence — each receives the output of the previous one.

```python
llmtrace.configure(
    enrichers=[enricher_a, enricher_b, enricher_c],
)
```

## Built-in Enrichers

### RedactPIIEnricher

Three-layer PII redaction:

1. **Key redaction** (normalizer) — redacts values for sensitive keys like `api_key`, `authorization` (see [Configuration](configuration.md#sensitive-key-redaction))
2. **Pattern redaction** (`PatternRedactor`) — regex-based detection of emails, credit cards, SSNs, etc.
3. **NLP redaction** (`PresidioRedactor`) — optional Microsoft Presidio integration for names, locations, medical data

```python
from llmtrace.transform.enrichment import RedactPIIEnricher, RedactionStrategy

enricher = RedactPIIEnricher(
    locales=("global", "en", "eu"),
    strategy=RedactionStrategy.REPLACE,
    use_presidio=False,
    custom_patterns={"EMPLOYEE_ID": r"EMP-\d{6}"},
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `locales` | `Sequence[str]` | `("global", "intl")` | Pattern locales to enable |
| `strategy` | `RedactionStrategy` | `REPLACE` | How to redact matched text |
| `use_presidio` | `bool` | `False` | Enable Presidio NLP engine |
| `presidio_language` | `str` | `"en"` | Language for Presidio analysis |
| `custom_patterns` | `dict[str, str] \| None` | `None` | Extra regex patterns (`name → regex`) |

#### Locales

| Locale | Detects |
|---|---|
| `"global"` | Email, credit card, IPv4/IPv6, URLs with auth, AWS keys, generic secrets |
| `"en"` | US phone, US SSN, US passport |
| `"eu"` | EU phone, IBAN, EU VAT number |
| `"intl"` | International phone, generic passport |

#### Redaction Strategies

| Strategy | Input | Output |
|---|---|---|
| `REPLACE` | `john@example.com` | `[EMAIL_REDACTED]` |
| `MASK` | `john@example.com` | `j***@*******.*om` |
| `HASH` | `john@example.com` | `[SHA:a1b2c3d4]` |

#### Presidio Integration

```bash
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg
```

```python
enricher = RedactPIIEnricher(use_presidio=True, presidio_language="en")
```

Default entities: `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, `LOCATION`, `MEDICAL_LICENSE`, `NRP`, `US_SSN`.

### CostAlertEnricher

Tags events exceeding a cost threshold.

```python
from decimal import Decimal
from llmtrace.transform.enrichment import CostAlertEnricher

enricher = CostAlertEnricher(threshold_usd=Decimal("0.50"))
```

Adds tag `cost_alert="high"` and metadata `cost_alert_threshold_usd` when `event.cost.total_cost > threshold`.

### LatencyClassifierEnricher

Classifies events by response time.

```python
from llmtrace.transform.enrichment import LatencyClassifierEnricher

enricher = LatencyClassifierEnricher(fast_ms=500, normal_ms=2000, slow_ms=5000)
```

| Latency | Classification |
|---|---|
| `< fast_ms` (500) | `"fast"` |
| `< normal_ms` (2000) | `"normal"` |
| `< slow_ms` (5000) | `"slow"` |
| `>= slow_ms` | `"critical"` |

Adds tag `latency_class` and metadata `latency_thresholds`.

### AddEnvironmentEnricher

Adds `hostname`, `pid`, `python_version`, `llmtrace_version` to metadata. No configuration needed.

## Writing a Custom Enricher

Any callable with signature `(TraceEvent) -> TraceEvent`:

```python
def add_user_context(event):
    event.tags["user_id"] = get_current_user_id()
    event.metadata["session_id"] = get_session_id()
    return event

llmtrace.configure(enrichers=[add_user_context])
```

## Writing a Custom Redactor

Implement `detect(text) -> list[RedactionMatch]`:

```python
from llmtrace.transform.enrichment import RedactionMatch

class InternalIDRedactor:
    _pattern = re.compile(r"EMP-\d{6}")

    def detect(self, text: str) -> list[RedactionMatch]:
        return [
            RedactionMatch(start=m.start(), end=m.end(), entity_type="EMPLOYEE_ID")
            for m in self._pattern.finditer(text)
        ]

    def __call__(self, text: str) -> list[RedactionMatch]:
        return self.detect(text)
```

Use with `RedactionEngine`:

```python
from llmtrace.transform.enrichment import RedactionEngine, RedactionStrategy

engine = RedactionEngine(
    redactors=[PatternRedactor(), InternalIDRedactor()],
    strategy=RedactionStrategy.HASH,
)
engine.redact("Contact EMP-123456 at john@example.com")
# "Contact [SHA:a1b2c3d4] at [SHA:e5f6g7h8]"
```

## Composing Enrichers

Order matters — enrichers run left to right:

```python
llmtrace.configure(
    enrichers=[
        RedactPIIEnricher(locales=("global", "en")),  # 1. redact PII first
        CostAlertEnricher(threshold_usd=Decimal("1.00")),  # 2. check cost
        LatencyClassifierEnricher(),  # 3. classify latency
        AddEnvironmentEnricher(),  # 4. add env info last
    ],
)
```

If any enricher raises, it's logged and skipped — the pipeline never crashes.

## EnrichmentPipeline

For standalone use outside `configure()`:

```python
from llmtrace.transform.enrichment import EnrichmentPipeline
pipeline = EnrichmentPipeline([enricher_a, enricher_b])
enriched_event = pipeline.apply(event)  # len(pipeline) == 2
```
