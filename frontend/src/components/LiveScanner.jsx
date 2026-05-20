import React, { useState } from 'react';
import axios from 'axios';

export default function LiveScanner() {
  const [logText, setLogText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!logText.trim()) return;
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/api/scan', { log_text: logText });
      setResult(response.data);
    } catch (err) {
      setResult({ error: err.message });
    }
    setLoading(false);
  };

  return (
    <div className="card">
      <h2 className="section-title">🔍 Live Scanner</h2>
      <textarea
        className="input-area"
        placeholder="Paste your telemetry log here..."
        value={logText}
        onChange={e => setLogText(e.target.value)}
        rows={8}
      />
      <button className="btn-primary" onClick={handleSubmit} disabled={loading}>
        {loading ? 'Scanning…' : 'Run Scan'}
      </button>
      {result && (
        <div className="result-panel">
          {result.error ? (
            <p className="error">Error: {result.error}</p>
          ) : (
            <>
              <p><strong>Decision:</strong> {result.decision}</p>
              <p><strong>Risk Level:</strong> {result.risk_level}</p>
              <p><strong>Saved USD:</strong> ${result.saved_usd.toFixed(4)}</p>
              <pre className="payload-preview">{result.processed_payload}</pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}
