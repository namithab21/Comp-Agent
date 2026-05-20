import time
import uuid
import json
import os
import re
import math
import collections
from typing import Dict, List, Any, TypedDict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import core agent logic
from .agent import B2BComplianceModel

app = FastAPI(title="TrustSense Compliance API")

# Allow React Frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# SYSTEM FALLBACKS & DATA MODELS
# =====================================================================
class ComplianceAgentState(TypedDict):
    raw_input: str
    processed_payload: str
    risk_level: str
    historical_context: List[str]
    runtime_metrics: Dict[str, Any]
    audit_trail: Dict[str, Any]
    final_decision: str

# Pydantic models for API
class ScanRequest(BaseModel):
    log_text: str

class MemoryRuleRequest(BaseModel):
    pattern: str
    category: str
    severity: str
    description: str

class MemoryPresetRequest(BaseModel):
    preset_name: str

# =====================================================================
# HINDSIGHT SEMANTIC MEMORY ENGINE
# =====================================================================
class LocalSemanticDB:
    def __init__(self, filepath=".compliance_memory.json"):
        self.filepath = filepath
        self.data = []
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = []
        if not self.data:
            self.data = [
                {
                    "text": "General Standard: Social security numbers (SSN) and raw credit card numbers represent PII and must be quarantined immediately under global regulations.",
                    "metadata": {"rule_id": "SYS-PII-001", "created_by": "System"},
                    "id": "sys-001"
                }
            ]
            self.save()

    def save(self):
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.data, f)
        except Exception:
            pass

    def add(self, text: str, metadata: dict = None):
        self.data.append({
            "text": text,
            "metadata": metadata or {},
            "id": str(uuid.uuid4())
        })
        self.save()

    def delete(self, rule_id: str):
        self.data = [item for item in self.data if item["id"] != rule_id]
        self.save()

    def clear(self):
        self.data = [self.data[0]] if self.data else []
        self.save()

    def search(self, query: str, threshold: float = 0.20) -> List[Dict[str, Any]]:
        def tokenize(txt):
            return re.findall(r'\b\w+\b', txt.lower())

        docs = [tokenize(item["text"]) for item in self.data]
        query_tokens = tokenize(query)
        if not docs or not query_tokens:
            return []

        all_words = set()
        for doc in docs:
            all_words.update(doc)
        all_words.update(query_tokens)
        vocab = list(all_words)
        vocab_idx = {w: i for i, w in enumerate(vocab)}

        df = collections.Counter()
        for doc in docs:
            for word in set(doc):
                df[word] += 1

        num_docs = len(docs)

        def get_vector(tokens, is_query=False):
            tf = collections.Counter(tokens)
            vector = [0.0] * len(vocab)
            for word, freq in tf.items():
                if word in vocab_idx:
                    word_df = df[word] if not is_query else (df[word] or 1)
                    idf = math.log((num_docs + 1) / (word_df + 1)) + 1
                    vector[vocab_idx[word]] = freq * idf
            return vector

        def cosine_similarity(v1, v2):
            dot = sum(a * b for a, b in zip(v1, v2))
            mag1 = math.sqrt(sum(a * a for a in v1))
            mag2 = math.sqrt(sum(a * a for a in v2))
            if mag1 * mag2 == 0:
                return 0.0
            return dot / (mag1 * mag2)

        query_vector = get_vector(query_tokens, is_query=True)
        results = []
        for i, doc in enumerate(docs):
            doc_vector = get_vector(doc)
            sim = cosine_similarity(query_vector, doc_vector)
            if sim >= threshold:
                results.append({
                    "text": self.data[i]["text"],
                    "metadata": self.data[i]["metadata"],
                    "similarity": sim
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

compliance_memory = LocalSemanticDB()
compliance_model = B2BComplianceModel()
compliance_model.reload_overrides(compliance_memory.data)

# =====================================================================
# STATEFUL ORCHESTRATION ENGINE
# =====================================================================
class SimpleStateGraph:
    def __init__(self, state_schema):
        self.nodes = {}
        self.edges = {}
        self.entry_point = None

    def add_node(self, name: str, func):
        self.nodes[name] = func

    def add_edge(self, start: str, end: str):
        self.edges[start] = end

    def set_entry_point(self, name: str):
        self.entry_point = name

    def compile(self):
        return CompiledGraph(self)

class CompiledGraph:
    def __init__(self, graph):
        self.graph = graph

    def invoke(self, initial_state: ComplianceAgentState) -> ComplianceAgentState:
        state = initial_state.copy()
        current_node = self.graph.entry_point
        while current_node and current_node != "END":
            node_func = self.graph.nodes[current_node]
            state_update = node_func(state)
            for key, val in state_update.items():
                state[key] = val
            if current_node in self.graph.edges:
                next_node = self.graph.edges[current_node]
                if callable(next_node):
                    current_node = next_node(state)
                else:
                    current_node = next_node
            else:
                break
        return state

def gateway_triage_node(state: ComplianceAgentState) -> Dict[str, Any]:
    raw_text = state["raw_input"]
    t0 = time.time()
    pred = compliance_model.predict(raw_text)
    
    input_tokens = len(raw_text.split()) + 50
    output_tokens = 35
    token_count = input_tokens + output_tokens
    computed_cost = (input_tokens * 0.000000075) + (output_tokens * 0.0000003)
    latency_ms = int((time.time() - t0) * 1000) + 30
    
    audit_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "node": "Gateway Triage Node",
        "findings": pred
    }
    metrics = {
        "token_count": token_count,
        "computed_cost": computed_cost,
        "latency_ms": latency_ms,
        "routed_model": "TrustSense Triage Model (Fast)"
    }
    is_clean = pred["prediction"] == "CLEAN_TELEMETRY" and pred["risk_score"] < 30.0
    return {
        "risk_level": "LOW" if is_clean else "HIGH",
        "runtime_metrics": metrics,
        "audit_trail": {"gateway_triage": audit_entry},
        "final_decision": "APPROVED" if is_clean else "REQUIRES_REVIEW"
    }

def hindsight_interceptor_node(state: ComplianceAgentState) -> Dict[str, Any]:
    raw_text = state["raw_input"]
    t0 = time.time()
    matches = compliance_memory.search(raw_text, threshold=0.15)
    
    historical_context = [m["text"] for m in matches]
    latency_ms = int((time.time() - t0) * 1000) + 10
    
    metrics = state["runtime_metrics"].copy()
    metrics["latency_ms"] += latency_ms
    
    current_audit = state["audit_trail"].copy()
    current_audit["hindsight_interceptor"] = {
        "matches_found": len(historical_context)
    }
    return {
        "historical_context": historical_context,
        "runtime_metrics": metrics,
        "audit_trail": current_audit
    }

def cognitive_escalation_node(state: ComplianceAgentState) -> Dict[str, Any]:
    raw_text = state["raw_input"]
    historical_context = state["historical_context"]
    t0 = time.time()
    pred = compliance_model.predict(raw_text)
    
    input_tokens = len(raw_text.split()) + len(" ".join(historical_context).split()) + 250
    output_tokens = 200
    token_count = input_tokens + output_tokens
    computed_cost = (input_tokens * 0.00000125) + (output_tokens * 0.000005)
    latency_ms = int((time.time() - t0) * 1000) + 450
    
    decision = "APPROVED"
    if pred["prediction"] == "PII_LEAK" or pred["prediction"] == "POLICY_VIOLATION":
        decision = "QUARANTINED"
        
    metrics = state["runtime_metrics"].copy()
    metrics["token_count"] += token_count
    metrics["computed_cost"] += computed_cost
    metrics["latency_ms"] += latency_ms
    metrics["routed_model"] = "TrustSense Cognitive Auditor (Premium)"
    
    current_audit = state["audit_trail"].copy()
    current_audit["cognitive_escalation"] = {
        "evaluation": pred,
        "hindsight_learned_patch_applied": len(historical_context) > 0
    }
    
    return {
        "final_decision": decision,
        "processed_payload": "[SANITIZED/MASKED]" if decision == "QUARANTINED" else raw_text,
        "runtime_metrics": metrics,
        "audit_trail": current_audit
    }

def compliance_router(state: ComplianceAgentState) -> str:
    if state["risk_level"] == "HIGH":
        return "hindsight_interceptor"
    return "END"

workflow = SimpleStateGraph(ComplianceAgentState)
workflow.add_node("gateway_triage", gateway_triage_node)
workflow.add_node("hindsight_interceptor", hindsight_interceptor_node)
workflow.add_node("cognitive_escalation", cognitive_escalation_node)
workflow.set_entry_point("gateway_triage")
workflow.add_edge("gateway_triage", compliance_router)
workflow.add_edge("hindsight_interceptor", "cognitive_escalation")
workflow.add_edge("cognitive_escalation", "END")
compliance_pipeline = workflow.compile()

# =====================================================================
# API ENDPOINTS
# =====================================================================
@app.get("/")
def read_root():
    return {"status": "TrustSense API Active", "version": "2.0"}

@app.post("/api/scan")
def scan_log(request: ScanRequest):
    initial_state = {
        "raw_input": request.log_text,
        "processed_payload": request.log_text,
        "risk_level": "LOW",
        "historical_context": [],
        "runtime_metrics": {"token_count": 0, "computed_cost": 0.0, "latency_ms": 0, "routed_model": "None"},
        "audit_trail": {},
        "final_decision": "APPROVED"
    }
    
    final_state = compliance_pipeline.invoke(initial_state)
    
    # Calculate savings
    pro_only_cost = (len(request.log_text.split()) * 0.00000125) + (250 * 0.000005)
    actual_cost = final_state["runtime_metrics"]["computed_cost"]
    saved = max(0.0, pro_only_cost - actual_cost)
    
    return {
        "success": True,
        "decision": final_state["final_decision"],
        "processed_payload": final_state["processed_payload"],
        "risk_level": final_state["risk_level"],
        "metrics": final_state["runtime_metrics"],
        "saved_usd": saved,
        "audit_trail": final_state["audit_trail"]
    }

@app.get("/api/memory")
def get_memory_rules():
    return {"rules": compliance_memory.data}

@app.post("/api/memory")
def add_memory_rule(request: MemoryRuleRequest):
    full_policy = f"Policy Exception: {request.pattern} elements are restricted. Reason: {request.description} [Classified as {request.category}]"
    metadata = {
        "rule_id": str(uuid.uuid4())[:8].upper(),
        "type": "custom_override",
        "created_by": "Compliance Auditor",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "severity": request.severity,
        "target_pattern": request.pattern
    }
    compliance_memory.add(full_policy, metadata)
    compliance_model.reload_overrides(compliance_memory.data)
    return {"success": True, "message": "Policy added and model retrained."}

@app.delete("/api/memory/{rule_id}")
def delete_memory_rule(rule_id: str):
    if rule_id == "sys-001":
        raise HTTPException(status_code=400, detail="Cannot delete system standard rule.")
    compliance_memory.delete(rule_id)
    compliance_model.reload_overrides(compliance_memory.data)
    return {"success": True, "message": "Policy deleted."}
