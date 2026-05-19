'use client'

type Layer = 'temperature' | 'deadzones' | 'planting'

interface NavTabsProps {
  active: Layer
  onChange: (l: Layer) => void
}

const tabs: { id: Layer; label: string; soon?: boolean }[] = [
  { id: 'temperature', label: 'Temperature' },
  { id: 'deadzones',   label: 'Dead Zones',  soon: true },
  { id: 'planting',    label: 'Planting Windows', soon: true },
]

export default function NavTabs({ active, onChange }: NavTabsProps) {
  return (
    <div style={{
      display: 'flex', gap: 4,
      background: 'var(--surface)',
      backdropFilter: 'var(--blur)',
      WebkitBackdropFilter: 'var(--blur)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
      padding: '4px',
      boxShadow: 'var(--shadow)',
    }}>
      {tabs.map(tab => (
        <button
          key={tab.id}
          onClick={() => !tab.soon && onChange(tab.id)}
          style={{
            padding: '7px 14px',
            borderRadius: 'var(--radius-md)',
            border: tab.id === active ? '1px solid rgba(59,130,246,0.25)' : '1px solid transparent',
            cursor: tab.soon ? 'default' : 'pointer',
            fontFamily: 'var(--font-display)',
            fontSize: 12,
            fontWeight: tab.id === active ? 600 : 400,
            background: tab.id === active ? 'rgba(59,130,246,0.15)' : 'transparent',
            color: tab.id === active ? '#93c5fd'
                 : tab.soon ? 'var(--text-muted)' : 'var(--text-secondary)',
            transition: 'all 0.15s',
            display: 'flex', alignItems: 'center', gap: 6,
          }}
        >
          {tab.label}
          {tab.soon && (
            <span style={{
              fontSize: 9, fontFamily: 'var(--font-mono)',
              color: 'var(--text-muted)', letterSpacing: '0.05em',
            }}>
              soon
            </span>
          )}
        </button>
      ))}
    </div>
  )
}