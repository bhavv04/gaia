"use client";

import { useState, useCallback } from "react";
import dynamic from "next/dynamic";

// D3 must be client-side only
const CascadeGraph = dynamic(() => import("./components/CascadeGraph"), { ssr: false });

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface NodeState {
  node_id: string;
  node_type: string;
  domain: string;
  region: string;
  latitude: number;
  longitude: number;
  current_value: number;
  confidence: number;
}

interface CascadeStep {
  step: number;
  cascading_nodes: string[];
  node_predictions: Record<string, number>;
  node_cascade_probs: Record<string, number>;
}

interface CascadeResponse {
  region: string;
  snapshot_id: string;
  nodes: NodeState[];
  edges: any[];
  cascade_sequence: CascadeStep[];
  fetch_time_ms: number;
  predict_time_ms: number;
  model_loaded: boolean;
}

interface D3Node {
  id: string;
  node_type: string;
  domain: string;
  value: number;
  confidence: number;
  cascading: boolean;
  cascade_prob: number;
}

// ---------------------------------------------------------------------------
// Preset regions
// ---------------------------------------------------------------------------

const PRESETS = [
  { label: "Gulf of Mexico",     lat: 29.0,  lon: -90.5, region: "Gulf of Mexico"     },
  { label: "Chesapeake Bay",     lat: 37.5,  lon: -76.1, region: "Chesapeake Bay"     },
  { label: "Baltic Sea",         lat: 57.0,  lon: 19.0,  region: "Baltic Sea"         },
  { label: "Mississippi Delta",  lat: 29.9,  lon: -89.9, region: "Mississippi Delta"  },
];

const DOMAIN_COLORS: Record<string, string> = {
  marine:       "#2dd4bf",
  atmospheric:  "#a78bfa",
  terrestrial:  "#4ade80",
  freshwater:   "#38bdf8",
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function Home() {
  const [lat, setLat]       = useState("29.0");
  const [lon, setLon]       = useState("-90.5");
  const [region, setRegion] = useState("Gulf of Mexico");
  const [steps, setSteps]   = useState(5);

  const [data, setData]           = useState<CascadeResponse | null>(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [hoveredNode, setHoveredNode] = useState<D3Node | null>(null);

  // ------------------------------------------------------------------
  // Fetch cascade prediction
  // ------------------------------------------------------------------
  const runPrediction = useCallback(async () => {
    setLoading(true);
    setError(null);
    setCurrentStep(0);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lat:    parseFloat(lat),
          lon:    parseFloat(lon),
          region: region,
          steps:  steps,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }

      const json: CascadeResponse = await res.json();
      setData(json);
    } catch (e: any) {
      setError(e.message ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [lat, lon, region, steps]);

  const applyPreset = (p: typeof PRESETS[0]) => {
    setLat(String(p.lat));
    setLon(String(p.lon));
    setRegion(p.region);
  };

  const currentStepData = data?.cascade_sequence[currentStep];

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------
  return (
    <main
      style={{
        minHeight:   "100vh",
        background:  "#060a0f",
        color:       "#e2e8f0",
        fontFamily:  "'DM Mono', 'Fira Code', monospace",
        display:     "flex",
        flexDirection:"column",
        gap:         "0",
      }}
    >
      {/* ---------------------------------------------------------------- */}
      {/* Header                                                            */}
      {/* ---------------------------------------------------------------- */}
      <header style={{
        borderBottom: "1px solid rgba(45,212,191,0.12)",
        padding:      "20px 32px",
        display:      "flex",
        alignItems:   "center",
        gap:          "16px",
      }}>
        {/* Animated pulse orb */}
        <div style={{ position: "relative", width: 12, height: 12 }}>
          <div style={{
            position:     "absolute",
            inset:        0,
            borderRadius: "50%",
            background:   "#2dd4bf",
            animation:    "pulse 2s ease-in-out infinite",
          }} />
        </div>

        <span style={{
          fontSize:      "13px",
          letterSpacing: "0.18em",
          color:         "#2dd4bf",
          textTransform: "uppercase",
        }}>
          GAIA
        </span>

        <span style={{
          fontSize:  "13px",
          color:     "rgba(255,255,255,0.25)",
          marginLeft:"4px",
        }}>
          ecological cascade prediction
        </span>

        {data && (
          <span style={{
            marginLeft:    "auto",
            fontSize:      "11px",
            color:         "rgba(255,255,255,0.3)",
            letterSpacing: "0.08em",
          }}>
            {data.snapshot_id} &nbsp;·&nbsp; {data.nodes.length} nodes &nbsp;·&nbsp;
            {data.edges.length} edges &nbsp;·&nbsp;
            {data.fetch_time_ms.toFixed(0)}ms fetch &nbsp;·&nbsp;
            {data.predict_time_ms.toFixed(0)}ms inference
            {!data.model_loaded && (
              <span style={{ color: "#f97316", marginLeft: 8 }}>
                ⚠ untrained weights
              </span>
            )}
          </span>
        )}
      </header>

      <div style={{ display: "flex", flex: 1, minHeight: 0 }}>

        {/* -------------------------------------------------------------- */}
        {/* Sidebar                                                          */}
        {/* -------------------------------------------------------------- */}
        <aside style={{
          width:        "280px",
          flexShrink:   0,
          borderRight:  "1px solid rgba(45,212,191,0.08)",
          padding:      "24px 20px",
          display:      "flex",
          flexDirection:"column",
          gap:          "24px",
          overflowY:    "auto",
        }}>

          {/* Presets */}
          <section>
            <label style={labelStyle}>REGION PRESETS</label>
            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              {PRESETS.map((p) => (
                <button
                  key={p.label}
                  onClick={() => applyPreset(p)}
                  style={{
                    ...presetBtnStyle,
                    borderColor: region === p.region
                      ? "rgba(45,212,191,0.5)"
                      : "rgba(255,255,255,0.06)",
                    color: region === p.region ? "#2dd4bf" : "rgba(255,255,255,0.5)",
                  }}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </section>

          {/* Manual coords */}
          <section>
            <label style={labelStyle}>COORDINATES</label>
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              <input
                value={region}
                onChange={(e) => setRegion(e.target.value)}
                placeholder="Region name"
                style={inputStyle}
              />
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
                <input
                  value={lat}
                  onChange={(e) => setLat(e.target.value)}
                  placeholder="Lat"
                  style={inputStyle}
                />
                <input
                  value={lon}
                  onChange={(e) => setLon(e.target.value)}
                  placeholder="Lon"
                  style={inputStyle}
                />
              </div>
            </div>
          </section>

          {/* Steps */}
          <section>
            <label style={labelStyle}>
              SIMULATION STEPS &nbsp;
              <span style={{ color: "#2dd4bf" }}>{steps}</span>
            </label>
            <input
              type="range"
              min={1} max={20}
              value={steps}
              onChange={(e) => setSteps(Number(e.target.value))}
              style={{ width: "100%", accentColor: "#2dd4bf" }}
            />
          </section>

          {/* Run button */}
          <button
            onClick={runPrediction}
            disabled={loading}
            style={{
              background:    loading ? "rgba(45,212,191,0.08)" : "rgba(45,212,191,0.12)",
              border:        "1px solid rgba(45,212,191,0.35)",
              borderRadius:  "6px",
              color:         loading ? "rgba(45,212,191,0.4)" : "#2dd4bf",
              padding:       "10px",
              fontSize:      "11px",
              letterSpacing: "0.14em",
              cursor:        loading ? "not-allowed" : "pointer",
              textTransform: "uppercase",
              transition:    "all 0.2s",
            }}
          >
            {loading ? "FETCHING DATA..." : "RUN PREDICTION"}
          </button>

          {error && (
            <div style={{
              fontSize:     "11px",
              color:        "#f87171",
              background:   "rgba(248,113,113,0.08)",
              border:       "1px solid rgba(248,113,113,0.2)",
              borderRadius: "6px",
              padding:      "8px 12px",
            }}>
              {error}
            </div>
          )}

          {/* Domain legend */}
          <section>
            <label style={labelStyle}>DOMAINS</label>
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              {Object.entries(DOMAIN_COLORS).map(([domain, color]) => (
                <div key={domain} style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <div style={{
                    width: 8, height: 8,
                    borderRadius: "50%",
                    background: color,
                    boxShadow: `0 0 6px ${color}`,
                  }} />
                  <span style={{ fontSize: "11px", color: "rgba(255,255,255,0.45)", textTransform: "capitalize" }}>
                    {domain}
                  </span>
                </div>
              ))}
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <div style={{
                  width: 8, height: 8,
                  borderRadius: "50%",
                  background: "#f97316",
                  boxShadow: "0 0 6px rgba(249,115,22,0.8)",
                }} />
                <span style={{ fontSize: "11px", color: "rgba(255,255,255,0.45)" }}>
                  cascading
                </span>
              </div>
            </div>
          </section>

          {/* Hovered node detail */}
          {hoveredNode && (
            <section style={{
              background:   "rgba(45,212,191,0.04)",
              border:       "1px solid rgba(45,212,191,0.12)",
              borderRadius: "8px",
              padding:      "12px",
            }}>
              <label style={{ ...labelStyle, color: DOMAIN_COLORS[hoveredNode.domain] ?? "#2dd4bf" }}>
                {hoveredNode.node_type.replace(/_/g, " ").toUpperCase()}
              </label>
              <div style={{ display: "flex", flexDirection: "column", gap: "4px", marginTop: "8px" }}>
                {[
                  ["domain",       hoveredNode.domain],
                  ["stress value", hoveredNode.value.toFixed(3)],
                  ["cascade prob", hoveredNode.cascade_prob.toFixed(3)],
                  ["confidence",   hoveredNode.confidence.toFixed(3)],
                  ["cascading",    hoveredNode.cascading ? "YES" : "no"],
                ].map(([k, v]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ fontSize: "10px", color: "rgba(255,255,255,0.3)", textTransform: "uppercase" }}>{k}</span>
                    <span style={{
                      fontSize: "10px",
                      color: k === "cascading" && v === "YES" ? "#f97316" : "rgba(255,255,255,0.7)",
                      fontWeight: k === "cascading" && v === "YES" ? "bold" : "normal",
                    }}>{v}</span>
                  </div>
                ))}
              </div>
            </section>
          )}
        </aside>

        {/* -------------------------------------------------------------- */}
        {/* Main canvas                                                       */}
        {/* -------------------------------------------------------------- */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

          {/* Graph */}
          <div style={{ flex: 1, padding: "24px", display: "flex", alignItems: "center", justifyContent: "center" }}>
            {!data && !loading && (
              <div style={{
                textAlign:     "center",
                color:         "rgba(255,255,255,0.15)",
                fontSize:      "13px",
                letterSpacing: "0.08em",
              }}>
                <div style={{ fontSize: "32px", marginBottom: "12px", opacity: 0.3 }}>⬡</div>
                Select a region and run a prediction
              </div>
            )}

            {loading && (
              <div style={{
                textAlign:     "center",
                color:         "#2dd4bf",
                fontSize:      "12px",
                letterSpacing: "0.12em",
                opacity:       0.7,
              }}>
                <div style={{ marginBottom: "12px", animation: "spin 1.5s linear infinite", display: "inline-block" }}>◌</div>
                <div>FETCHING ECOLOGICAL DATA</div>
                <div style={{ marginTop: "6px", color: "rgba(255,255,255,0.3)", fontSize: "10px" }}>
                  NOAA · OpenAQ · NASA Earthdata · iNaturalist
                </div>
              </div>
            )}

            {data && !loading && (
              <CascadeGraph
                data={data}
                currentStep={currentStep}
                width={Math.min(900, window.innerWidth - 360)}
                height={Math.min(560, window.innerHeight - 220)}
                onNodeHover={setHoveredNode}
              />
            )}
          </div>

          {/* Step scrubber */}
          {data && (
            <div style={{
              borderTop:    "1px solid rgba(45,212,191,0.08)",
              padding:      "16px 32px",
              display:      "flex",
              alignItems:   "center",
              gap:          "20px",
            }}>
              <span style={{ fontSize: "10px", color: "rgba(255,255,255,0.3)", letterSpacing: "0.1em", whiteSpace: "nowrap" }}>
                CASCADE STEP
              </span>

              <div style={{ display: "flex", gap: "6px" }}>
                {data.cascade_sequence.map((s, i) => {
                  const hasCascade = s.cascading_nodes.length > 0;
                  return (
                    <button
                      key={i}
                      onClick={() => setCurrentStep(i)}
                      style={{
                        width:        "32px",
                        height:       "32px",
                        borderRadius: "6px",
                        border:       `1px solid ${i === currentStep
                          ? (hasCascade ? "rgba(249,115,22,0.7)" : "rgba(45,212,191,0.5)")
                          : "rgba(255,255,255,0.08)"}`,
                        background: i === currentStep
                          ? (hasCascade ? "rgba(249,115,22,0.15)" : "rgba(45,212,191,0.1)")
                          : "transparent",
                        color: i === currentStep
                          ? (hasCascade ? "#f97316" : "#2dd4bf")
                          : "rgba(255,255,255,0.3)",
                        fontSize:  "11px",
                        cursor:    "pointer",
                        fontFamily:"'DM Mono', monospace",
                        position:  "relative",
                      }}
                    >
                      {i + 1}
                      {hasCascade && (
                        <div style={{
                          position:     "absolute",
                          top:          "3px",
                          right:        "3px",
                          width:        "4px",
                          height:       "4px",
                          borderRadius: "50%",
                          background:   "#f97316",
                          boxShadow:    "0 0 4px #f97316",
                        }} />
                      )}
                    </button>
                  );
                })}
              </div>

              {currentStepData && currentStepData.cascading_nodes.length > 0 && (
                <div style={{
                  marginLeft:   "8px",
                  fontSize:     "11px",
                  color:        "#f97316",
                  letterSpacing:"0.06em",
                }}>
                  {currentStepData.cascading_nodes.length} node{currentStepData.cascading_nodes.length > 1 ? "s" : ""} cascading
                  &nbsp;·&nbsp;
                  {currentStepData.cascading_nodes.map(n => n.split("_").slice(0, 2).join(" ")).join(", ")}
                </div>
              )}

              {currentStepData && currentStepData.cascading_nodes.length === 0 && (
                <div style={{ fontSize: "11px", color: "rgba(255,255,255,0.2)", letterSpacing: "0.06em" }}>
                  no cascade at this step
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&display=swap');

        * { box-sizing: border-box; margin: 0; padding: 0; }

        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 0 0 rgba(45,212,191,0.4); }
          50%       { opacity: 0.7; transform: scale(1.2); box-shadow: 0 0 0 6px rgba(45,212,191,0); }
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }

        input[type="range"] {
          -webkit-appearance: none;
          height: 2px;
          background: rgba(45,212,191,0.2);
          border-radius: 2px;
          outline: none;
        }
        input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 12px; height: 12px;
          border-radius: 50%;
          background: #2dd4bf;
          box-shadow: 0 0 8px rgba(45,212,191,0.6);
          cursor: pointer;
        }

        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(45,212,191,0.2); border-radius: 2px; }
      `}</style>
    </main>
  );
}

// ---------------------------------------------------------------------------
// Style constants
// ---------------------------------------------------------------------------

const labelStyle: React.CSSProperties = {
  fontSize:      "10px",
  letterSpacing: "0.14em",
  color:         "rgba(255,255,255,0.25)",
  textTransform: "uppercase",
  display:       "block",
  marginBottom:  "8px",
};

const inputStyle: React.CSSProperties = {
  width:        "100%",
  background:   "rgba(255,255,255,0.04)",
  border:       "1px solid rgba(255,255,255,0.08)",
  borderRadius: "6px",
  padding:      "7px 10px",
  color:        "rgba(255,255,255,0.7)",
  fontSize:     "12px",
  fontFamily:   "'DM Mono', monospace",
  outline:      "none",
};

const presetBtnStyle: React.CSSProperties = {
  background:    "transparent",
  border:        "1px solid",
  borderRadius:  "6px",
  padding:       "7px 10px",
  fontSize:      "11px",
  letterSpacing: "0.05em",
  cursor:        "pointer",
  textAlign:     "left",
  fontFamily:    "'DM Mono', monospace",
  transition:    "all 0.15s",
};