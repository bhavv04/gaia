'use client'

interface SidebarProps {
  year: number
  onYearChange: (y: number) => void
  showAnomalies: boolean
  onToggleAnomalies: () => void
  loading: boolean
}

const glass: React.CSSProperties = {
  background: 'var(--surface)',
  backdropFilter: 'var(--blur)',
  WebkitBackdropFilter: 'var(--blur)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-lg)',
  boxShadow: 'var(--shadow)',
}

export default function Sidebar({
  year, onYearChange, showAnomalies, onToggleAnomalies, loading
}: SidebarProps) {
  return (
    <div style={{ ...glass, padding: '16px', height: '100%', display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Logo */}
      <div>
        <div style={{ fontFamily: 'var(--font-display)', fontSize: 20, fontWeight: 700, letterSpacing: '-0.5px', color: 'var(--text-primary)' }}>
          gaia
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)', letterSpacing: '0.12em', textTransform: 'uppercase', marginTop: 2 }}>
          climate explorer
        </div>
      </div>

      <div style={{ width: '100%', height: '1px', background: 'var(--border)' }} />

      {/* Dataset */}
      <div>
        <Label>dataset</Label>
        <div style={selectStyle}>ERA5 Reanalysis</div>
      </div>

      {/* Variable */}
      <div>
        <Label>variable</Label>
        <div style={selectStyle}>Surface temp anomaly</div>
      </div>

      {/* Year */}
      <div>
        <Label>year</Label>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>1950</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 22, fontWeight: 300, color: 'var(--text-primary)', letterSpacing: '-1px' }}>
            {year}
          </span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>2023</span>
        </div>
        <input
          type="range"
          min={1950}
          max={2023}
          value={year}
          onChange={e => onYearChange(Number(e.target.value))}
          style={{ width: '100%', accentColor: 'var(--accent)', cursor: 'pointer' }}
        />
      </div>

      {/* Layers */}
      <div>
        <Label>layers</Label>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <LayerRow
            color="#ef4444"
            label="Temp anomaly"
            active={true}
            onToggle={() => {}}
          />
          <LayerRow
            color="#3b82f6"
            label="ML anomalies"
            active={showAnomalies}
            onToggle={onToggleAnomalies}
          />
          <LayerRow
            color="#22c55e"
            label="Dead zones"
            active={false}
            onToggle={() => {}}
            disabled
          />
        </div>
      </div>

      <div style={{ marginTop: 'auto' }}>
        <div style={{ width: '100%', height: '1px', background: 'var(--border)', marginBottom: 14 }} />
        <Label>ml model</Label>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          Isolation Forest<br />
          <span style={{ color: 'var(--text-muted)' }}>contamination 3%</span>
        </div>
        {loading && (
          <div style={{
            marginTop: 10, fontFamily: 'var(--font-mono)', fontSize: 10,
            color: 'var(--accent)', letterSpacing: '0.05em',
          }}>
            loading {year}...
          </div>
        )}
      </div>

    </div>
  )
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)',
      textTransform: 'uppercase', letterSpacing: '0.12em', marginBottom: 8,
    }}>
      {children}
    </div>
  )
}

const selectStyle: React.CSSProperties = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-sm)',
  padding: '7px 10px',
  fontSize: 12,
  color: 'var(--text-secondary)',
  fontFamily: 'var(--font-display)',
}

function LayerRow({ color, label, active, onToggle, disabled = false }: {
  color: string; label: string; active: boolean;
  onToggle: () => void; disabled?: boolean
}) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8,
      opacity: disabled ? 0.35 : 1,
    }}>
      <div style={{ width: 7, height: 7, borderRadius: '50%', background: color, flexShrink: 0 }} />
      <span style={{ fontSize: 12, color: 'var(--text-secondary)', fontFamily: 'var(--font-display)', flex: 1 }}>
        {label}
      </span>
      <button
        onClick={disabled ? undefined : onToggle}
        style={{
          width: 28, height: 16, borderRadius: 8,
          background: active ? 'var(--accent)' : 'var(--border)',
          border: 'none', cursor: disabled ? 'default' : 'pointer',
          position: 'relative', transition: 'background 0.2s',
          flexShrink: 0,
        }}
      >
        <div style={{
          position: 'absolute', top: 2,
          left: active ? 14 : 2,
          width: 12, height: 12, borderRadius: '50%',
          background: '#fff', transition: 'left 0.2s',
        }} />
      </button>
    </div>
  )
}