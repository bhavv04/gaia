'use client'

interface StatsPanelProps {
  year: number
  yearSummary: { mean_anomaly: number; max_anomaly: number; min_anomaly: number } | undefined
  anomalyData: { anomaly_count: number; features: any[] } | null
}

const glass: React.CSSProperties = {
  background: 'var(--surface)',
  backdropFilter: 'var(--blur)',
  WebkitBackdropFilter: 'var(--blur)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-lg)',
  boxShadow: 'var(--shadow)',
}

export default function StatsPanel({ year, yearSummary, anomalyData }: StatsPanelProps) {
  const mean = yearSummary?.mean_anomaly
  const max = yearSummary?.max_anomaly
  const anomalyCount = anomalyData?.anomaly_count ?? 0

  const top5 = anomalyData?.features.slice(0, 5) ?? []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, height: '100%' }}>

      {/* Global mean */}
      <div style={{ ...glass, padding: '14px 16px' }}>
        <Label>global mean anomaly</Label>
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: 28, fontWeight: 300,
          letterSpacing: '-1px', lineHeight: 1,
          color: mean == null ? 'var(--text-muted)'
               : mean > 0 ? '#ef4444' : '#3b82f6',
        }}>
          {mean == null ? '—' : `${mean > 0 ? '+' : ''}${mean.toFixed(2)}°C`}
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
          vs 1951–1980 baseline
        </div>
      </div>

      {/* Max anomaly */}
      <div style={{ ...glass, padding: '14px 16px' }}>
        <Label>peak anomaly</Label>
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: 20, fontWeight: 300,
          color: '#f97316', letterSpacing: '-0.5px',
        }}>
          {max == null ? '—' : `+${max.toFixed(2)}°C`}
        </div>
      </div>

      {/* ML anomaly count */}
      <div style={{ ...glass, padding: '14px 16px' }}>
        <Label>ml flagged</Label>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: 28, fontWeight: 300,
            color: anomalyCount > 0 ? '#ef4444' : 'var(--text-muted)',
            letterSpacing: '-1px', lineHeight: 1,
          }}>
            {anomalyCount}
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>
            regions
          </div>
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
          isolation forest · 3% threshold
        </div>
      </div>

      {/* Top anomalies */}
      {top5.length > 0 && (
        <div style={{ ...glass, padding: '14px 16px', flex: 1, overflow: 'hidden' }}>
          <Label>top anomalies</Label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {top5.map((f, i) => {
              const { anomaly_c } = f.properties
              const [lon, lat] = f.geometry.coordinates
              const isWarm = anomaly_c > 0
              const pct = Math.min(Math.abs(anomaly_c) / 5, 1) * 100
              return (
                <div key={i}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>
                      {lat.toFixed(0)}°, {lon.toFixed(0)}°
                    </span>
                    <span style={{
                      fontFamily: 'var(--font-mono)', fontSize: 10,
                      color: isWarm ? '#ef4444' : '#3b82f6',
                    }}>
                      {anomaly_c > 0 ? '+' : ''}{anomaly_c.toFixed(1)}°
                    </span>
                  </div>
                  <div style={{ height: 2, background: 'var(--border)', borderRadius: 1 }}>
                    <div style={{
                      height: '100%', width: `${pct}%`,
                      background: isWarm ? '#ef4444' : '#3b82f6',
                      borderRadius: 1, transition: 'width 0.4s ease',
                    }} />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

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