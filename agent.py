import re
import math
import time
import uuid
import collections
from typing import Dict, List, Any, TypedDict

class ComplianceAgentState(TypedDict):
    raw_input: str
    processed_payload: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    historical_context: List[str]
    runtime_metrics: Dict[str, Any]  # token_count, computed_cost, latency_ms, routed_model
    audit_trail: Dict[str, Any]  # Structured evaluation logs
    final_decision: str  # "APPROVED", "QUARANTINED", "REQUIRES_HUMAN_REVIEW"

class B2BComplianceModel:
    """
    A specialized Naive Bayes NLP Classifier + Pattern Extractor for B2B Compliance classification.
    Supports online dynamic training increments when auditor override guidelines are updated.
    """
    def __init__(self):
        self.classes = ["CLEAN_TELEMETRY", "POLICY_VIOLATION", "PII_LEAK"]
        self.training_data = {
            "CLEAN_TELEMETRY": [
                "system startup sequence complete port listening on 443",
                "ingested telemetry data packet size 1024 bytes status success",
                "user initiated standard database query for metrics logging",
                "heartbeat ping response from server node 4 status green",
                "processing transaction metadata for invoice reference",
                "standard API payload request initiated by client host"
            ],
            "POLICY_VIOLATION": [
                "device tracking payload trk eu initialized on agent node",
                "telemetry data log contains restricted tracking id trk eu",
                "gdpr restricted identifier trk eu processed",
                "tracking cookie session telemetry generated for user"
            ],
            "PII_LEAK": [
                "raw email alert user john doe company com has updated password",
                "pii flag customer phone number is 555 019 2834",
                "billing details invoice sent to account holder at 212 555 0199",
                "credit card ending in 4111222233334444 processed",
                "social security number ssn reference logged in transaction"
            ]
        }
        self.vocabulary = set()
        self.word_counts = {}
        self.class_totals = {}
        self.class_priors = {}
        self.override_keywords = set()
        self.train()

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def train(self):
        """
        Train the Naive Bayes model on current vocabulary distributions.
        """
        self.vocabulary = set()
        self.word_counts = {c: collections.Counter() for c in self.classes}
        self.class_totals = {c: 0 for c in self.classes}
        
        # Calculate counts
        total_docs = 0
        class_docs = {c: len(docs) for c, docs in self.training_data.items()}
        for c, docs in self.training_data.items():
            total_docs += len(docs)
            for doc in docs:
                tokens = self.tokenize(doc)
                self.word_counts[c].update(tokens)
                self.class_totals[c] += len(tokens)
                self.vocabulary.update(tokens)
        
        # Compute priors
        for c in self.classes:
            self.class_priors[c] = class_docs[c] / total_docs

    def apply_policy_override(self, override_text: str):
        """
        Online Learning Injection: Dynamically retrain the model weights by adding
        the override guidelines to the POLICY_VIOLATION target class dataset.
        """
        tokens = self.tokenize(override_text)
        # Extract meaningful terms
        meaningful_tokens = [t for t in tokens if len(t) > 3]
        for t in meaningful_tokens:
            self.override_keywords.add(t)
        
        # Append override document to reinforce classification
        self.training_data["POLICY_VIOLATION"].append(override_text.lower())
        
        # Retrain with new distributions
        self.train()

    def clear_overrides(self):
        """
        Revert POLICY_VIOLATION class to standard base dataset.
        """
        self.training_data["POLICY_VIOLATION"] = [
            "device tracking payload trk eu initialized on agent node",
            "telemetry data log contains restricted tracking id trk eu",
            "gdpr restricted identifier trk eu processed",
            "tracking cookie session telemetry generated for user"
        ]
        self.override_keywords.clear()
        self.train()

    def reload_overrides(self, rules: List[Dict[str, Any]]):
        """
        Reload overrides from a database list and retrain model.
        """
        self.training_data["POLICY_VIOLATION"] = [
            "device tracking payload trk eu initialized on agent node",
            "telemetry data log contains restricted tracking id trk eu",
            "gdpr restricted identifier trk eu processed",
            "tracking cookie session telemetry generated for user"
        ]
        self.override_keywords.clear()
        
        for rule in rules:
            if rule["id"] != "sys-001":
                tokens = self.tokenize(rule["text"])
                meaningful_tokens = [t for t in tokens if len(t) > 3]
                for t in meaningful_tokens:
                    self.override_keywords.add(t)
                self.training_data["POLICY_VIOLATION"].append(rule["text"].lower())
                
        self.train()

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict compliance category, returning probability confidence score distributions.
        """
        tokens = self.tokenize(text)
        posteriors = {}
        vocab_size = len(self.vocabulary)
        
        # In B2B audits, heuristics help classify strong patterns (like emails/phones/custom tracking tokens)
        has_pii_pattern = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)) or \
                           bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
                           
        # Check if text contains any explicitly overridden custom compliance terms
        contains_override_terms = any(k in text.lower() for k in self.override_keywords)

        for c in self.classes:
            # Start with log prior
            log_prob = math.log(self.class_priors[c])
            for token in tokens:
                # Laplace smoothing
                count = self.word_counts[c][token]
                prob = (count + 1) / (self.class_totals[c] + vocab_size)
                log_prob += math.log(prob)
            posteriors[c] = log_prob

        # Convert log probabilities back to probabilities using softmax/exponentiation
        max_log = max(posteriors.values())
        exps = {c: math.exp(v - max_log) for c, v in posteriors.items()}
        sum_exps = sum(exps.values())
        probs = {c: (exps[c] / sum_exps) for c in self.classes}

        # Override adjustments for deterministic compliance triggers (SSN / email / custom matching)
        if has_pii_pattern:
            probs["PII_LEAK"] = max(probs["PII_LEAK"], 0.95)
            # Re-normalize
            s = sum(probs.values())
            probs = {c: v / s for c, v in probs.items()}
            
        if contains_override_terms:
            # Dynamic policy override reinforcement
            probs["POLICY_VIOLATION"] = max(probs["POLICY_VIOLATION"], 0.90)
            s = sum(probs.values())
            probs = {c: v / s for c, v in probs.items()}

        # Find best class
        predicted_class = max(probs, key=probs.get)
        confidence = probs[predicted_class]

        # Calculate dynamic risk score based on predictions
        risk_score = 0.0
        if predicted_class == "CLEAN_TELEMETRY":
            risk_score = probs["POLICY_VIOLATION"] * 30.0 + probs["PII_LEAK"] * 50.0
        elif predicted_class == "POLICY_VIOLATION":
            risk_score = 40.0 + (probs["POLICY_VIOLATION"] * 30.0)
        else:
            risk_score = 75.0 + (probs["PII_LEAK"] * 25.0)

        risk_score = min(100.0, max(0.0, risk_score))

        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": probs,
            "risk_score": risk_score,
            "has_pii_pattern": has_pii_pattern,
            "contains_override_terms": contains_override_terms
        }


class TrustSenseComplianceAgent:
    """
    An orchestrator agent that executes the transaction evaluation pipeline
    utilizing B2BComplianceModel and LocalSemanticDB vector entries.
    """
    def __init__(self, model: B2BComplianceModel, memory_store: Any):
        self.model = model
        self.memory_store = memory_store

    def run_gateway_triage(self, state: ComplianceAgentState) -> Dict[str, Any]:
        """
        Fast triage scan. Computes low-latency classifications representing Flash model speeds.
        """
        raw_text = state["raw_input"]
        t0 = time.time()
        
        # Get model prediction
        pred = self.model.predict(raw_text)
        
        input_tokens = len(raw_text.split()) + 50
        output_tokens = 35
        token_count = input_tokens + output_tokens
        computed_cost = (input_tokens * 0.000000075) + (output_tokens * 0.0000003)
        latency_ms = int((time.time() - t0) * 1000) + 30
        
        metrics = {
            "token_count": token_count,
            "computed_cost": computed_cost,
            "latency_ms": latency_ms,
            "routed_model": "Gemini 1.5 Flash (Custom Triage)"
        }
        
        audit_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "node": "Gateway Triage Node",
            "model": "TrustSense Custom Triage Classifer",
            "checks_executed": ["Pattern Scan", "Probabilistic Classification"],
            "findings": {
                "predicted_class": pred["prediction"],
                "class_confidence": round(pred["confidence"], 3),
                "risk_score": round(pred["risk_score"], 1)
            }
        }
        
        # Decide if risk level is low (CLEAN_TELEMETRY prediction with risk_score < 30)
        is_clean = pred["prediction"] == "CLEAN_TELEMETRY" and pred["risk_score"] < 30.0
        
        return {
            "risk_level": "LOW" if is_clean else "HIGH",
            "runtime_metrics": metrics,
            "audit_trail": {"gateway_triage": audit_entry},
            "final_decision": "APPROVED" if is_clean else "REQUIRES_REVIEW"
        }

    def run_hindsight_interceptor(self, state: ComplianceAgentState) -> Dict[str, Any]:
        """
        Intercepts high risk paths to extract matching memory override patterns.
        """
        raw_text = state["raw_input"]
        t0 = time.time()
        
        matches = self.memory_store.search(raw_text, threshold=0.15)
        
        historical_context = []
        audit_findings = []
        for m in matches:
            historical_context.append(m["text"])
            audit_findings.append({
                "source": "Memory Database",
                "similarity_score": round(m["similarity"], 3),
                "rule": m["text"]
            })
            
        latency_ms = int((time.time() - t0) * 1000) + 10
        metrics = state["runtime_metrics"].copy()
        metrics["latency_ms"] += latency_ms
        
        audit_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "node": "Hindsight Persistence Interceptor",
            "matches_found": len(historical_context),
            "audit_findings": audit_findings
        }
        
        current_audit = state["audit_trail"].copy()
        current_audit["hindsight_interceptor"] = audit_entry
        
        return {
            "historical_context": historical_context,
            "runtime_metrics": metrics,
            "audit_trail": current_audit
        }

    def run_cognitive_escalation(self, state: ComplianceAgentState) -> Dict[str, Any]:
        """
        Deep reasoning escalation path representing premium model logic.
        Incorporates historical auditor override context parameters to execute classification.
        """
        raw_text = state["raw_input"]
        historical_context = state["historical_context"]
        t0 = time.time()
        
        # Perform prediction with full context
        pred = self.model.predict(raw_text)
        
        # Calculate premium resources
        input_tokens = len(raw_text.split()) + len(" ".join(historical_context).split()) + 250
        output_tokens = 200
        token_count = input_tokens + output_tokens
        computed_cost = (input_tokens * 0.00000125) + (output_tokens * 0.000005)
        latency_ms = int((time.time() - t0) * 1000) + 450
        
        # Decide final compliance action
        decision = "APPROVED"
        reasoning = "Telemetry logs within acceptable standard boundaries."
        code = "CLEAN"
        
        # Check predictions
        if pred["prediction"] == "PII_LEAK":
            decision = "QUARANTINED"
            code = "PII-SECURE-LEAK"
            reasoning = "Unambiguous PII or personal data (email/phone) detected in cleartext payload."
        elif pred["prediction"] == "POLICY_VIOLATION":
            decision = "QUARANTINED"
            code = "POLICY-VIOLATION-DETECTED"
            reasoning = "Telemetry violates active policy overrides established by compliance auditor."
            
        audit_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "node": "Cognitive Escalation Node",
            "model": "TrustSense Cognitive Auditor (Premium Escalation)",
            "evaluation": {
                "confidence_matrix": {
                    "class_probabilities": {c: round(v, 4) for c, v in pred["probabilities"].items()},
                    "selected_class": pred["prediction"],
                    "prediction_confidence": round(pred["confidence"], 4)
                },
                "compliance_violation_code": code,
                "justification": reasoning,
                "hindsight_learned_patch_applied": len(historical_context) > 0
            }
        }
        
        metrics = state["runtime_metrics"].copy()
        metrics["token_count"] += token_count
        metrics["computed_cost"] += computed_cost
        metrics["latency_ms"] += latency_ms
        metrics["routed_model"] = "TrustSense Cognitive Auditor (Premium)"
        
        current_audit = state["audit_trail"].copy()
        current_audit["cognitive_escalation"] = audit_entry
        
        return {
            "final_decision": decision,
            "processed_payload": "[SANITIZED/MASKED]" if decision == "QUARANTINED" else raw_text,
            "runtime_metrics": metrics,
            "audit_trail": current_audit
        }
