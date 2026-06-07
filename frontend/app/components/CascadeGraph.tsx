"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import * as d3 from "d3";

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

interface EdgeState {
  source: string;
  target: string;
  edge_type: string;
  weight: number;
  lag_hours: number;
  cross_domain: boolean;
}

interface CascadeStep {
  step: number;
  cascading_nodes: string[];
  node_predictions: Record<string, number>;
  node_cascade_probs: Record<string, GLfloat>;
}

interface CascadeResponse { 
  region: string;
  snapshot_id: string;
  nodes: NodeState[];
  edges: EdgeState[];
  cascade_sequence: CascadeStep[];
  fetch_time_ms: number;
  predict_time_ms: number;
  model_loaded: boolean;
}

interface D3Node extends d3.SimulationNodeDatum {
  id: string;
  node_type: string;
  domain: string;
  value: number;
  confidence: number;
  cascading: boolean;
  cascade_prob: number;
}

interface D3Link extends d3.SimulationLinkDatum<D3Node> {
  edge_type: string;
  weight: number;
  lag_hours: number;
  cross_domain: boolean;
}

// ---------------------------------------------------------------------------
// Domain colour palette — bioluminescent by domain
// ---------------------------------------------------------------------------

const DOMAIN_COLORS: Record<string, string> = {
  marine:       "#2dd4bf",  // teal glow
  atmospheric:  "#a78bfa",  // violet
  terrestrial:  "#4ade80",  // green
  freshwater:   "#38bdf8",  // sky blue
};

const DOMAIN_GLOW: Record<string, string> = {
  marine:       "rgba(45,212,191,0.35)",
  atmospheric:  "rgba(167,139,250,0.35)",
  terrestrial:  "rgba(74,222,128,0.35)",
  freshwater:   "rgba(56,189,248,0.35)",
};

const CASCADE_COLOR  = "#f97316";  // amber — cascade warning
const CASCADE_GLOW   = "rgba(249,115,22,0.5)";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface CascadeGraphProps {
  data:          CascadeResponse | null;
  currentStep:   number;
  width?:        number;
  height?:       number;
  onNodeHover?:  (node: D3Node | null) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CascadeGraph({
  data,
  currentStep,
  width  = 900,
  height = 600,
  onNodeHover,
}: CascadeGraphProps) {
  const svgRef        = useRef<SVGSVGElement>(null);
  const simRef        = useRef<d3.Simulation<D3Node, D3Link> | null>(null);
  const nodesRef      = useRef<D3Node[]>([]);
  const linksRef      = useRef<D3Link[]>([]);

  // ------------------------------------------------------------------
  // Build D3 node/link arrays from API response
  // ------------------------------------------------------------------
  const buildGraph = useCallback((resp: CascadeResponse, step: number) => {
    const stepData = resp.cascade_sequence[step] ?? resp.cascade_sequence[0];
    const cascading = new Set(stepData?.cascading_nodes ?? []);

    const nodes: D3Node[] = resp.nodes.map((n) => ({
      id:           n.node_id,
      node_type:    n.node_type,
      domain:       n.domain,
      value:        stepData?.node_predictions[n.node_id] ?? n.current_value,
      confidence:   n.confidence,
      cascading:    cascading.has(n.node_id),
      cascade_prob: stepData?.node_cascade_probs[n.node_id] ?? 0,
    }));

    const nodeSet = new Set(nodes.map((n) => n.id));
    const links: D3Link[] = resp.edges
      .filter((e) => nodeSet.has(e.source) && nodeSet.has(e.target))
      .map((e) => ({
        source:      e.source,
        target:      e.target,
        edge_type:   e.edge_type,
        weight:      e.weight,
        lag_hours:   e.lag_hours,
        cross_domain:e.cross_domain,
      }));

    return { nodes, links };
  }, []);

  // ------------------------------------------------------------------
  // Initialise D3 simulation
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!data || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const { nodes, links } = buildGraph(data, currentStep);
    nodesRef.current = nodes;
    linksRef.current = links;

    // Defs — filters for glow effects
    const defs = svg.append("defs");

    Object.entries(DOMAIN_COLORS).forEach(([domain, color]) => {
      const filter = defs.append("filter")
        .attr("id", `glow-${domain}`)
        .attr("x", "-50%").attr("y", "-50%")
        .attr("width", "200%").attr("height", "200%");
      filter.append("feGaussianBlur")
        .attr("stdDeviation", "4")
        .attr("result", "blur");
      filter.append("feFlood")
        .attr("flood-color", DOMAIN_GLOW[domain])
        .attr("result", "color");
      filter.append("feComposite")
        .attr("in", "color").attr("in2", "blur").attr("operator", "in")
        .attr("result", "glow");
      const merge = filter.append("feMerge");
      merge.append("feMergeNode").attr("in", "glow");
      merge.append("feMergeNode").attr("in", "SourceGraphic");
    });

    // Cascade glow filter
    const cascadeFilter = defs.append("filter")
      .attr("id", "glow-cascade")
      .attr("x", "-80%").attr("y", "-80%")
      .attr("width", "260%").attr("height", "260%");
    cascadeFilter.append("feGaussianBlur").attr("stdDeviation", "8").attr("result", "blur");
    cascadeFilter.append("feFlood").attr("flood-color", CASCADE_GLOW).attr("result", "color");
    cascadeFilter.append("feComposite")
      .attr("in", "color").attr("in2", "blur").attr("operator", "in").attr("result", "glow");
    const cm = cascadeFilter.append("feMerge");
    cm.append("feMergeNode").attr("in", "glow");
    cm.append("feMergeNode").attr("in", "SourceGraphic");

    // Arrow markers
    defs.append("marker")
      .attr("id", "arrow-default")
      .attr("viewBox", "0 -4 8 8")
      .attr("refX", 18).attr("refY", 0)
      .attr("markerWidth", 5).attr("markerHeight", 5)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-4L8,0L0,4")
      .attr("fill", "rgba(255,255,255,0.15)");

    defs.append("marker")
      .attr("id", "arrow-cross")
      .attr("viewBox", "0 -4 8 8")
      .attr("refX", 18).attr("refY", 0)
      .attr("markerWidth", 5).attr("markerHeight", 5)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-4L8,0L0,4")
      .attr("fill", "rgba(249,115,22,0.6)");

    // Background grid — subtle bioluminescent effect
    const gridGroup = svg.append("g").attr("class", "grid").attr("opacity", 0.04);
    for (let x = 0; x < width; x += 40) {
      gridGroup.append("line")
        .attr("x1", x).attr("y1", 0).attr("x2", x).attr("y2", height)
        .attr("stroke", "#2dd4bf").attr("stroke-width", 0.5);
    }
    for (let y = 0; y < height; y += 40) {
      gridGroup.append("line")
        .attr("x1", 0).attr("y1", y).attr("x2", width).attr("y2", y)
        .attr("stroke", "#2dd4bf").attr("stroke-width", 0.5);
    }

    // Link layer
    const linkGroup = svg.append("g").attr("class", "links");
    const linkSel = linkGroup
      .selectAll<SVGLineElement, D3Link>("line")
      .data(links)
      .join("line")
      .attr("stroke", (d) => d.cross_domain ? "rgba(249,115,22,0.5)" : "rgba(255,255,255,0.12)")
      .attr("stroke-width", (d) => Math.max(0.5, d.weight * 2))
      .attr("stroke-dasharray", (d) => d.cross_domain ? "4 3" : "none")
      .attr("marker-end", (d) => d.cross_domain ? "url(#arrow-cross)" : "url(#arrow-default)");

    // Node layer
    const nodeGroup = svg.append("g").attr("class", "nodes");
    const nodeSel = nodeGroup
      .selectAll<SVGGElement, D3Node>("g")
      .data(nodes, (d) => d.id)
      .join("g")
      .attr("class", "node")
      .style("cursor", "pointer")
      .call(
        d3.drag<SVGGElement, D3Node>()
          .on("start", (event, d) => {
            if (!event.active) simRef.current?.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on("end", (event, d) => {
            if (!event.active) simRef.current?.alphaTarget(0);
            d.fx = null; d.fy = null;
          })
      );

    // Outer pulse ring for cascading nodes
    nodeSel.append("circle")
      .attr("class", "pulse-ring")
      .attr("r", (d) => d.cascading ? 22 : 0)
      .attr("fill", "none")
      .attr("stroke", CASCADE_COLOR)
      .attr("stroke-width", 1.5)
      .attr("opacity", 0.6);

    // Value fill — inner circle radius encodes stress level
    nodeSel.append("circle")
      .attr("class", "value-fill")
      .attr("r", (d) => 6 + d.value * 10)
      .attr("fill", (d) => d.cascading
        ? CASCADE_COLOR
        : DOMAIN_COLORS[d.domain] ?? "#888"
      )
      .attr("opacity", (d) => 0.15 + d.value * 0.35)
      .attr("filter", (d) => d.cascading
        ? "url(#glow-cascade)"
        : `url(#glow-${d.domain})`
      );

    // Core node circle
    nodeSel.append("circle")
      .attr("class", "node-core")
      .attr("r", 8)
      .attr("fill", (d) => d.cascading ? CASCADE_COLOR : DOMAIN_COLORS[d.domain] ?? "#888")
      .attr("stroke", "rgba(255,255,255,0.15)")
      .attr("stroke-width", 1)
      .attr("filter", (d) => d.cascading
        ? "url(#glow-cascade)"
        : `url(#glow-${d.domain})`
      );

    // Node label
    nodeSel.append("text")
      .attr("dy", 22)
      .attr("text-anchor", "middle")
      .attr("fill", "rgba(255,255,255,0.55)")
      .attr("font-size", "9px")
      .attr("font-family", "'DM Mono', 'Fira Code', monospace")
      .attr("letter-spacing", "0.03em")
      .text((d) => d.node_type.replace(/_/g, " "));

    // Hover interactions
    nodeSel
      .on("mouseenter", (_, d) => onNodeHover?.(d))
      .on("mouseleave", () => onNodeHover?.(null));

    // Force simulation
    const sim = d3.forceSimulation<D3Node>(nodes)
      .force("link", d3.forceLink<D3Node, D3Link>(links)
        .id((d) => d.id)
        .distance((d) => d.cross_domain ? 180 : 120)
        .strength((d) => d.weight * 0.4)
      )
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide(28));

    simRef.current = sim;

    sim.on("tick", () => {
      linkSel
        .attr("x1", (d) => (d.source as D3Node).x ?? 0)
        .attr("y1", (d) => (d.source as D3Node).y ?? 0)
        .attr("x2", (d) => (d.target as D3Node).x ?? 0)
        .attr("y2", (d) => (d.target as D3Node).y ?? 0);

      nodeSel.attr("transform", (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    return () => { sim.stop(); };
  }, [data, buildGraph]);

  // ------------------------------------------------------------------
  // Update node appearance when cascade step changes (no re-simulation)
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!data || !svgRef.current) return;
    const stepData = data.cascade_sequence[currentStep];
    if (!stepData) return;

    const cascading = new Set(stepData.cascading_nodes);
    const svg       = d3.select(svgRef.current);

    svg.selectAll<SVGGElement, D3Node>(".node").each(function(d) {
      const isCascading = cascading.has(d.id);
      d.cascading   = isCascading;
      d.value       = stepData.node_predictions[d.id] ?? d.value;
      d.cascade_prob= stepData.node_cascade_probs[d.id] ?? d.cascade_prob;

      const g = d3.select(this);

      g.select(".node-core")
        .transition().duration(600).ease(d3.easeCubicOut)
        .attr("fill", isCascading ? CASCADE_COLOR : DOMAIN_COLORS[d.domain] ?? "#888")
        .attr("filter", isCascading ? "url(#glow-cascade)" : `url(#glow-${d.domain})`);

      g.select(".value-fill")
        .transition().duration(600).ease(d3.easeCubicOut)
        .attr("r", 6 + d.value * 10)
        .attr("fill", isCascading ? CASCADE_COLOR : DOMAIN_COLORS[d.domain] ?? "#888")
        .attr("opacity", 0.15 + d.value * 0.35);

      g.select(".pulse-ring")
        .transition().duration(300)
        .attr("r", isCascading ? 22 : 0)
        .attr("opacity", isCascading ? 0.6 : 0);

      // Pulse animation on cascade entry
      if (isCascading) {
        g.select(".pulse-ring")
          .attr("r", 22)
          .transition().duration(800).ease(d3.easeExpOut)
          .attr("r", 32).attr("opacity", 0)
          .transition().duration(0)
          .attr("r", 22).attr("opacity", 0.6);
      }
    });
  }, [currentStep, data]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      style={{
        background: "radial-gradient(ellipse at 40% 50%, #0d1117 0%, #060a0f 100%)",
        borderRadius: "12px",
        border: "1px solid rgba(45,212,191,0.12)",
      }}
    />
  );
}