'use client'

import { useState, useEffect, useRef } from 'react'

interface TimeScrubberProps {
  year: number
  onChange: (y: number) => void
  summary: Array<{ year: number; mean_anomaly: number }>
  loading: boolean
}

export default function TimeScrubber({ year, onChange, summary, loading }: TimeScrubberProps) {
  const [playing, setPlaying] = useState(false)
  const intervalRef = useRef<number | null>(null)

  useEffect(() => {
    if (playing) {
      intervalRef.current = window.setInterval(() => {
        const next = year + 1
        if (next > 2023) { setPlaying(false); return }
        onChange(next)
      }, 800)
    } else {
      if (intervalRef.current !== null) window.clearInterval(intervalRef.current)
    }
    return () => { if (intervalRef.current !== null) window.clearInterval(intervalRef.current) }
  }, [playing, year, onChange])

  // Normalize anomaly for bar height
  const maxAbs = Math.max(...summary.map(s => Math.abs(s.mean_anomaly)), 1)

  return (
    <div style={{
      background: 'var(--surface)',
      backdropFilter: 'var(--blur)',
      WebkitBackdropFilter: 'var(--blur)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
      boxShadow: 'var(--shadow)',
      padding: '12px 16px',
      display: 'flex',
      alignItems: 'center',
      gap: 14,
      minWidth: 480,
    }}>

      {/* Play/pause */}
      <button
        onClick={() => setPlaying(v => !v)}
        style={{
          width: 30, height: 30, borderRadius: '50%',
          background: playing ? 'var(--accent)' : 'rgba(255,255,255,0.06)',
          border: '1px solid var(--border-light)',
          cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexShrink: 0, transition: 'background 0.2s',
        }}
      >
        {playing ? (
          // Pause icon
          <div style={{ display: 'flex', gap: 3 }}>
            <div style={{ width: 3, height: 10, background: '#fff', borderRadius: 1 }} />
            <div style={{ width: 3, height: 10, background: '#fff', borderRadius: 1 }} />
          </div>
        ) : (
          // Play icon
          <div style={{
            width: 0, height: 0,
            borderTop: '5px solid transparent',
            borderBottom: '5px solid transparent',
            borderLeft: '9px solid #fff',
            marginLeft: 2,
          }} />
        )}
      </button>

      {/* Mini sparkline bars */}
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 32, flex: 1 }}>
        {summary.map(s => {
          const isActive = s.year === year
          const height = Math.max((Math.abs(s.mean_anomaly) / maxAbs) * 28, 3)
          const color = s.mean_anomaly > 0 ? '#ef4444' : '#3b82f6'
          return (
            <div
              key={s.year}
              onClick={() => onChange(s.year)}
              title={`${s.year}: ${s.mean_anomaly > 0 ? '+' : ''}${s.mean_anomaly.toFixed(2)}°C`}
              style={{
                flex: 1,
                height,
                background: isActive ? color : `${color}44`,
                borderRadius: '2px 2px 0 0',
                cursor: 'pointer',
                transition: 'height 0.3s ease, background 0.2s',
                minWidth: 2,
                outline: isActive ? `1px solid ${color}` : 'none',
              }}
            />
          )
        })}
      </div>

      {/* Year label */}
      <div style={{
        fontFamily: 'var(--font-mono)', fontSize: 18, fontWeight: 300,
        color: loading ? 'var(--text-muted)' : 'var(--text-primary)',
        letterSpacing: '-0.5px', minWidth: 44, textAlign: 'right',
        transition: 'color 0.2s',
      }}>
        {year}
      </div>

    </div>
  )
}