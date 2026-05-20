import React, { useState } from 'react';
import './index.css';
import { LayoutDashboard, ShieldAlert, Library, Activity } from 'lucide-react';
import LiveScanner from './components/LiveScanner';
import PolicyManager from './components/PolicyManager';
import AuditVault from './components/AuditVault';

function App() {
  const [activeTab, setActiveTab] = useState('scanner');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Top Header */}
      <header className="glass-panel" style={{ margin: '20px', padding: '16px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ background: 'linear-gradient(135deg, #10b981 0%, #3b82f6 100%)', width: '36px', height: '36px', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '18px' }}>
            🛡️
          </div>
          <h1 style={{ fontSize: '1.4rem', fontWeight: 800 }} className="text-gradient">TrustSense</h1>
          <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginLeft: '8px', borderLeft: '1px solid var(--border-light)', paddingLeft: '16px' }}>Enterprise Data Security Engine</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem', color: 'var(--success)', background: 'rgba(16,185,129,0.1)', padding: '6px 12px', borderRadius: '20px', border: '1px solid rgba(16,185,129,0.2)' }}>
          <span className="live-indicator"></span> Cascadeflow Active
        </div>
      </header>

      <div style={{ display: 'flex', flexGrow: 1, padding: '0 20px 20px 20px', gap: '20px', overflow: 'hidden' }}>
        {/* Sidebar Navigation */}
        <nav className="glass-panel" style={{ width: '250px', padding: '20px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <p style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '8px' }}>Workspace</p>
          
          <button onClick={() => setActiveTab('scanner')} className="btn" style={{ justifyContent: 'flex-start', background: activeTab === 'scanner' ? 'rgba(59,130,246,0.15)' : 'transparent', border: 'none', color: activeTab === 'scanner' ? '#fff' : 'var(--text-muted)' }}>
            <Activity size={18} color={activeTab === 'scanner' ? '#60a5fa' : 'currentColor'} /> Live Scanner
          </button>
          
          <button onClick={() => setActiveTab('policy')} className="btn" style={{ justifyContent: 'flex-start', background: activeTab === 'policy' ? 'rgba(59,130,246,0.15)' : 'transparent', border: 'none', color: activeTab === 'policy' ? '#fff' : 'var(--text-muted)' }}>
            <ShieldAlert size={18} color={activeTab === 'policy' ? '#60a5fa' : 'currentColor'} /> Policy Engine
          </button>
          
          <button onClick={() => setActiveTab('vault')} className="btn" style={{ justifyContent: 'flex-start', background: activeTab === 'vault' ? 'rgba(59,130,246,0.15)' : 'transparent', border: 'none', color: activeTab === 'vault' ? '#fff' : 'var(--text-muted)' }}>
            <Library size={18} color={activeTab === 'vault' ? '#60a5fa' : 'currentColor'} /> Audit Vault
          </button>
          
          <div style={{ marginTop: 'auto' }}>
             <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textAlign: 'center' }}>Agentic AI Hackathon</p>
          </div>
        </nav>

        {/* Main Content Area */}
        <main className="glass-panel" style={{ flexGrow: 1, overflowY: 'auto', padding: '30px' }}>
          {activeTab === 'scanner' && <LiveScanner />}
          {activeTab === 'policy' && <PolicyManager />}
          {activeTab === 'vault' && <AuditVault />}
        </main>
      </div>
    </div>
  );
}

export default App;
