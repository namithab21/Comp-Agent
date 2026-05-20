import React, { useState, useEffect } from 'react';
import axios from 'axios';

export default function PolicyManager() {
  const [policies, setPolicies] = useState([]);
  const [newPolicy, setNewPolicy] = useState({ pattern: '', category: '', severity: '', description: '' });
  const [loading, setLoading] = useState(false);

  const fetchPolicies = async () => {
    try {
      const res = await axios.get('http://localhost:8000/api/memory');
      setPolicies(res.data.rules);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchPolicies();
  }, []);

  const handleAdd = async () => {
    if (!newPolicy.pattern) return;
    setLoading(true);
    try {
      await axios.post('http://localhost:8000/api/memory', newPolicy);
      setNewPolicy({ pattern: '', category: '', severity: '', description: '' });
      fetchPolicies();
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const handleDelete = async (id) => {
    try {
      await axios.delete(`http://localhost:8000/api/memory/${id}`);
      fetchPolicies();
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="card">
      <h2 className="section-title">🛡️ Policy Engine</h2>
      <div className="policy-list">
        {policies.map(p => (
          <div key={p.id} className="policy-item">
            <p>{p.text}</p>
            <button className="btn-danger" onClick={() => handleDelete(p.id)}>Delete</button>
          </div>
        ))}
      </div>
      <div className="policy-form">
        <h3>Add New Policy</h3>
        <input placeholder="Pattern" value={newPolicy.pattern} onChange={e => setNewPolicy({ ...newPolicy, pattern: e.target.value })} />
        <input placeholder="Category" value={newPolicy.category} onChange={e => setNewPolicy({ ...newPolicy, category: e.target.value })} />
        <input placeholder="Severity" value={newPolicy.severity} onChange={e => setNewPolicy({ ...newPolicy, severity: e.target.value })} />
        <textarea placeholder="Description" value={newPolicy.description} onChange={e => setNewPolicy({ ...newPolicy, description: e.target.value })} />
        <button className="btn-primary" onClick={handleAdd} disabled={loading}>Add Policy</button>
      </div>
    </div>
  );
}
