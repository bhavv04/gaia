'use client'

import { useState, useEffect } from 'react'
import dynamic from 'next/dynamic'
import Sidebar from '@/components/sidebar/SideBar'
import StatsPanel from '@/components/panels/StatsPanel'
import TimeScrubber from '@/components/map/TimeScrubber'
import NavTabs from '@/components/ui/NavTabs'

// Leaflet must be loaded client-side only
const ClimateMap = dynamic(() => import('@/components/map/ClimateMap'), { ssr: false })

type Layer = 'temperature' | 'deadzones' | 'planting'

export default function Home() {
  const [ready, setReady] = useState(false)
  const [splashDone, setSplashDone] = useState(false)
  const [year, setYear] = useState(2014)
  const [activeLayer, setActiveLayer] = useState<Layer>('temperature')
  const [showAnomalies, setShowAnomalies] = useState(true)
  const [geoData, setGeoData] = useState<any>(null)
  const [anomalyData, setAnomalyData] = useState<any>(null)
  const [summary, setSummary] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  // Load summary once
  useEffect(() => {
    fetch('http://localhost:8000/api/temperature/summary')
      .then(r => r.json())
      .then(d => setSummary(d.summary))
      .catch(console.error)
  }, [])

  // Load data when year changes
  useEffect(() => {
    setLoading(true)
    const requests = [
      fetch(`http://localhost:8000/api/temperature?year=${year}`).then(r => r.json()),
      fetch(`http://localhost:8000/api/anomalies?year=${year}`).then(r => r.json()),
    ]
    Promise.all(requests)
      .then(([geo, anom]) => {
        setGeoData(geo)
        setAnomalyData(anom)
        setLoading(false)
        setReady(true)
      })
      .catch(e => { console.error(e); setLoading(false) })
  }, [year])

  // Splash screen timing
  useEffect(() => {
    if (ready && !splashDone) {
      setTimeout(() => setSplashDone(true), 800)
    }
  }, [ready])

  const yearSummary = summary.find(s => s.year === year)

  return (
    <main style={{ position: 'relative', width: '100vw', height: '100vh', overflow: 'hidden' }}>

      {/* ── Splash ── */}
      {!splashDone && (
        <div style={{
          position: 'absolute', inset: 0, zIndex: 100,
          background: '#080c12',
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center', gap: 24,
          transition: 'opacity 0.6s ease',
          opacity: ready ? 0 : 1,
          pointerEvents: ready ? 'none' : 'all',
        }}>
          <div style={{
            fontFamily: 'var(--font-display)', fontSize: 56,
            fontWeight: 700, letterSpacing: '-2px', color: '#e8edf5',
          }}>
            gaia
          </div>
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: 11,
            color: 'var(--text-muted)', letterSpacing: '0.15em',
            textTransform: 'uppercase',
          }}>
            climate explorer
          </div>
          <div style={{
            width: 120, height: 2, background: 'var(--border)',
            borderRadius: 1, overflow: 'hidden', marginTop: 8,
          }}>
            <div style={{
              height: '100%', borderRadius: 1,
              background: 'linear-gradient(90deg, transparent, var(--accent), transparent)',
              backgroundSize: '200% 100%',
              animation: 'shimmer 1.2s ease infinite',
            }} />
          </div>
        </div>
      )}

      {/* ── Map ── */}
      <ClimateMap
        geoData={geoData}
        anomalyData={showAnomalies ? anomalyData : null}
      />

      {/* ── Nav tabs (top center) ── */}
      <div style={{
        position: 'absolute', top: 20, left: '50%',
        transform: 'translateX(-50%)', zIndex: 10,
      }}>
        <NavTabs active={activeLayer} onChange={setActiveLayer} />
      </div>

      {/* ── Sidebar (left) ── */}
      <div style={{
        position: 'absolute', top: 20, left: 20, bottom: 80,
        zIndex: 10, width: 220,
        animation: splashDone ? 'fadeIn 0.5s ease both' : 'none',
      }}>
        <Sidebar
          year={year}
          onYearChange={setYear}
          showAnomalies={showAnomalies}
          onToggleAnomalies={() => setShowAnomalies(v => !v)}
          loading={loading}
        />
      </div>

      {/* ── Stats panel (right) ── */}
      <div style={{
        position: 'absolute', top: 20, right: 20, bottom: 80,
        zIndex: 10, width: 200,
        animation: splashDone ? 'fadeIn 0.5s ease 0.1s both' : 'none',
      }}>
        <StatsPanel
          year={year}
          yearSummary={yearSummary}
          anomalyData={anomalyData}
        />
      </div>

      {/* ── Time scrubber (bottom center) ── */}
      <div style={{
        position: 'absolute', bottom: 24, left: '50%',
        transform: 'translateX(-50%)', zIndex: 10,
        animation: splashDone ? 'fadeIn 0.5s ease 0.2s both' : 'none',
      }}>
        <TimeScrubber
          year={year}
          onChange={setYear}
          summary={summary}
          loading={loading}
        />
      </div>

    </main>
  )
}