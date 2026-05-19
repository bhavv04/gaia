'use client'

import { useEffect, useRef } from 'react'
import { MapContainer, TileLayer, useMap, Popup } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// ── Color scale for anomaly values ──────────────────────────────────────────
function anomalyToColor(val: number): string {
  if (val <= -3)   return '#1d4ed8'
  if (val <= -2)   return '#3b82f6'
  if (val <= -1)   return '#93c5fd'
  if (val <= -0.3) return '#cbd5e1'
  if (val <= 0.3)  return '#334155'
  if (val <= 1)    return '#fbbf24'
  if (val <= 2)    return '#f97316'
  if (val <= 3)    return '#ef4444'
  return '#7f1d1d'
}

function anomalyOpacity(val: number): number {
  const abs = Math.abs(val)
  if (abs < 0.3) return 0.15
  if (abs < 1)   return 0.35
  if (abs < 2)   return 0.55
  return 0.75
}

// ── Grid cell layer using canvas for performance ─────────────────────────────
interface GeoData {
  features: Array<{
    geometry: { coordinates: [number, number] }
    properties: { temp_c: number; anomaly_c: number }
  }>
}

function HeatLayer({ geoData }: { geoData: GeoData | null }) {
  const map = useMap()
  const layerRef = useRef<L.LayerGroup | null>(null)

  useEffect(() => {
    if (!geoData) return

    if (layerRef.current) {
      map.removeLayer(layerRef.current)
    }

    const group = L.layerGroup()

    // Render grid cells as small rectangles
    for (const feature of geoData.features) {
      const [lon, lat] = feature.geometry.coordinates
      const { anomaly_c, temp_c } = feature.properties
      const color = anomalyToColor(anomaly_c)
      const opacity = anomalyOpacity(anomaly_c)

      // Each cell is 2°×2° (matches our coarsened grid)
      const bounds: L.LatLngBoundsExpression = [
        [lat - 1, lon - 1],
        [lat + 1, lon + 1],
      ]

      L.rectangle(bounds, {
        color: 'transparent',
        fillColor: color,
        fillOpacity: opacity,
        weight: 0,
      }).addTo(group)
    }

    group.addTo(map)
    layerRef.current = group

    return () => {
      map.removeLayer(group)
    }
  }, [geoData, map])

  return null
}

// ── Anomaly marker layer ─────────────────────────────────────────────────────
interface AnomalyData {
  features: Array<{
    geometry: { coordinates: [number, number] }
    properties: { anomaly_c: number; temp_c: number; anomaly_score: number }
  }>
}

function AnomalyMarkers({ anomalyData }: { anomalyData: AnomalyData | null }) {
  const map = useMap()
  const layerRef = useRef<L.LayerGroup | null>(null)

  useEffect(() => {
    if (layerRef.current) map.removeLayer(layerRef.current)
    if (!anomalyData) return

    const group = L.layerGroup()

    // Show top 12 most extreme anomalies only
    const top = anomalyData.features.slice(0, 12)

    for (const feature of top) {
      const [lon, lat] = feature.geometry.coordinates
      const { anomaly_c, temp_c } = feature.properties
      const isWarm = anomaly_c > 0
      const color = isWarm ? '#ef4444' : '#3b82f6'

      const icon = L.divIcon({
        className: '',
        html: `
          <div style="position:relative;width:20px;height:20px;">
            <div style="
              position:absolute;inset:0;
              border-radius:50%;
              border:1.5px solid ${color};
              animation:pulse-ring 1.8s ease-out infinite;
            "></div>
            <div style="
              position:absolute;top:50%;left:50%;
              transform:translate(-50%,-50%);
              width:7px;height:7px;
              border-radius:50%;
              background:${color};
              box-shadow:0 0 6px ${color};
            "></div>
          </div>
        `,
        iconSize: [20, 20],
        iconAnchor: [10, 10],
      })

      const popupContent = `
        <div style="
          padding:12px 14px;min-width:160px;
          font-family:'Syne',sans-serif;
        ">
          <div style="
            font-size:10px;letter-spacing:0.1em;text-transform:uppercase;
            color:${color};margin-bottom:8px;font-weight:600;
          ">anomaly detected</div>
          <div style="font-size:20px;font-weight:700;color:#e8edf5;line-height:1;">
            ${anomaly_c > 0 ? '+' : ''}${anomaly_c.toFixed(2)}°C
          </div>
          <div style="font-size:11px;color:#7a8599;margin-top:4px;">
            above 1951–1980 baseline
          </div>
          <div style="
            margin-top:10px;padding-top:10px;
            border-top:1px solid rgba(255,255,255,0.07);
            font-size:11px;color:#7a8599;font-family:'DM Mono',monospace;
          ">
            ${lat.toFixed(1)}°, ${lon.toFixed(1)}° · ${temp_c.toFixed(1)}°C
          </div>
        </div>
      `

      L.marker([lat, lon], { icon })
        .bindPopup(popupContent, { maxWidth: 220, minWidth: 180 })
        .addTo(group)
    }

    group.addTo(map)
    layerRef.current = group

    return () => { map.removeLayer(group) }
  }, [anomalyData, map])

  return null
}

// ── Main map component ───────────────────────────────────────────────────────
interface ClimateMapProps {
  geoData: GeoData | null
  anomalyData: AnomalyData | null
}

export default function ClimateMap({ geoData, anomalyData }: ClimateMapProps) {
  return (
    <MapContainer
      center={[20, 0]}
      zoom={2.5}
      minZoom={2}
      maxZoom={6}
      style={{ width: '100%', height: '100%', position: 'absolute', inset: 0 }}
      zoomControl={false}
      attributionControl={true}
    >
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/dark_matter_nolabels/{z}/{x}/{y}{r}.png"
        attribution='&copy; <a href="https://carto.com">CARTO</a>'
      />
      <HeatLayer geoData={geoData} />
      <AnomalyMarkers anomalyData={anomalyData} />
    </MapContainer>
  )
}