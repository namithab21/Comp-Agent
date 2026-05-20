import React, { useEffect, useState } from 'react';
import axios from 'axios';

export default function AuditVault() {
  const [auditData, setAuditData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAudit = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/memory');
        setAuditData(response.data.rules || []);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchAudit();
  }, []);

  return (
    <div className="card">
      <h2 className="section-title">📂 Audit Vault</h2>
      {loading && <p>Loading audit rules...</p>}
      {error && <p className="error">Error: {error}</p>}
      {!loading && !error && (
        <ul className="audit-list">
          {auditData.map((item, idx) => (
            <li key={item.id || idx} className="audit-item">
              <strong>{item.metadata?.rule_id || 'Rule'}:</strong> {item.text}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
