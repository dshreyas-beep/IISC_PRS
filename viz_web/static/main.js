// ═══════════════════════════════════════════════════════════════════════════
//  WildSim · main.js · IISc Centre for Ecological Sciences
//  Agent-Based Ecological Telemetry — Professional Research Build
// ═══════════════════════════════════════════════════════════════════════════

let ws = null;
let world = null;
let zones = [];
let topo = null;
let agents = [];
let hotspots = [];
let graph = null;
let waterResources = [];

// Display toggles
let showZones = true;
let showIntent = true;
let showTrails = true;
let showLabels = true;
let show3D = true;

// Forensic Tool states
let showGhostTrails = true;
let showProbe = false;
let probeFilter = "all";
let probeTargetGrid = null; // Stored as {x, y} in tensor grid space
let selectedAgentId = null;

const tooltip = document.getElementById("tip");

const canvas = document.getElementById("sim");
const ctx = canvas?.getContext("2d");
const canvas3d = document.getElementById("sim3d");

// Bannerghatta bbox + grid mapping for agent coordinates.
const GRID_W = 240;
const GRID_H = 720;
const NORTH = 12.8229972;
const SOUTH = 12.3443525;
const EAST = 77.6368603;
const WEST = 77.4828815;
const CENTER_LAT = 12.8008;
const CENTER_LON = 77.5756;

function gridToLatLng(x, y) {
  const lon = WEST + (Number(x) / GRID_W) * (EAST - WEST);
  const lat = NORTH - (Number(y) / GRID_H) * (NORTH - SOUTH);
  return [lat, lon];
}

function latLonToGrid(lat, lon, gridW = GRID_W, gridH = GRID_H) {
  const x = ((Number(lon) - WEST) / (EAST - WEST)) * Number(gridW);
  const y = ((NORTH - Number(lat)) / (NORTH - SOUTH)) * Number(gridH);
  return [x, y];
}

const bnpLandmarks = [
  { name: "Kanakapura Wildlife Underpass", lat: 12.7650, lon: 77.5150, type: "bridge" },
  { name: "Suvarnamukhi Stream Corridor", lat: 12.8000, lon: 77.5700, type: "water" },
  { name: "Jigani Industrial Border", lat: 12.7800, lon: 77.6300, type: "village" },
  { name: "Tattekere Elephant Camp", lat: 12.7100, lon: 77.5200, type: "water" },
];

function markerColorForSpecies(species) {
  species = (species || "").toLowerCase();
  if (species === "elephant") return "#9aa1a8"; 
  if (species === "leopard") return "#f1c40f"; 
  if (species === "sloth_bear") return "#27ae60"; 
  if (species === "tiger") return "#9c3b24"; 
  if (species === "lion") return "#8f6a2d"; 
  if (species === "human") return "#1e3a5f"; 
  return "#6b6b62";
}

function drawMapLandmarks(ctx, cellW, cellH) {
  if (!ctx || !world) return;
  const dpr = devicePixelRatio || 1;

  ctx.save();
  ctx.font = `700 ${11 * dpr}px Arial`;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";

  for (const lm of bnpLandmarks) {
    const [gx, gy] = latLonToGrid(lm.lat, lm.lon);
    const px = (gx * cellW);
    const py = (gy * cellH);

    const sxp = sx(px, py);
    const syp = sy(px, py);

    if (lm.type === "bridge") {
      const w = 10 * dpr;
      const h = 10 * dpr;
      ctx.fillStyle = "rgba(192, 57, 43, 0.92)";
      ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
      ctx.lineWidth = 2 * dpr;
      ctx.fillRect(sxp - w / 2, syp - h / 2, w, h);
      ctx.strokeRect(sxp - w / 2, syp - h / 2, w, h);
    }

    const labelX = sxp + 8 * dpr;
    const labelY = syp - 10 * dpr;
    ctx.lineWidth = 3.5 * dpr;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.92)";
    ctx.strokeText(lm.name, labelX, labelY);
    ctx.fillStyle = "rgba(20, 20, 18, 0.95)";
    ctx.fillText(lm.name, labelX, labelY);
  }

  ctx.restore();
}

let scale = 1, offsetX = 0, offsetY = 0;
let rotateMap = false; 

const trails = new Map();
const MAX_TRAIL = 80; // Extended length for Ghost Trails analysis

let mouseX = 0, mouseY = 0;
let hoveredAgent = null;
let hoveredHotspot = null;

let cachedLayerBytes = null;
let cachedLayerKey   = "";
let bgCanvas         = null;

let pendingReset    = false;
let simTime         = 0;
let frameCount      = 0;
let prevAgentStates = new Map();

// ─── DESIGN TOKENS ──────────────────────────────────────
const T = {
  paper:      "#f7f5f0",
  paper1:     "#f0ede6",
  paper2:     "#e8e4db",
  ink:        "#1a1a18",
  ink2:       "#3d3d38",
  ink3:       "#6b6b62",
  ink4:       "#9a9a8e",
  rule:       "#d4d0c8",
  ruleLight:  "#e4e0d8",
  accent:     "#2c5f8a",

  elephant:   "#6b4c11",   
  leopard:    "#8b2020",   
  bear:       "#3d5a3e",   
  human:      "#1e3a5f",   

  ok:         "#2e6b3e",
  warn:       "#8b4513",
};

function fitCanvas() {
  if (!canvas || !ctx) return;
  const scroller = document.getElementById("cw");
  const rect = (scroller || canvas.parentElement || canvas).getBoundingClientRect();
  const dpr = devicePixelRatio || 1;
  const w = Math.max(1, Math.floor(rect.width * dpr));
  const h = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }

  if (canvas3d) {
    if (canvas3d.width !== w || canvas3d.height !== h) {
      canvas3d.width = w;
      canvas3d.height = h;
    }
  }

  if (world) {
    rotateMap = world.H > world.W * 1.25 && rect.width > rect.height;
    const viewW = rotateMap ? world.H : world.W;
    const viewH = rotateMap ? world.W : world.H;
    const sxv = canvas.width / Math.max(1, viewW);
    const syv = canvas.height / Math.max(1, viewH);
    scale = Math.max(0.25, Math.min(sxv, syv));
    offsetX = (canvas.width - viewW * scale) * 0.5;
    offsetY = (canvas.height - viewH * scale) * 0.5;
  }
}
window.addEventListener("resize", fitCanvas);

function applyViewMode() {
  const badge = document.getElementById("viewModeBadge");
  if (badge) badge.textContent = "GRID VIEW";
}

// Convert tensor space to screen space
function sx(x, y) {
  if (!world) return offsetX + x * scale;
  return rotateMap ? (offsetX + (world.H - y) * scale) : (offsetX + x * scale);
}
function sy(x, y) {
  if (!world) return offsetY + y * scale;
  return rotateMap ? (offsetY + x * scale) : (offsetY + y * scale);
}

// NEW: Convert screen space back to tensor space (for the Probe click)
function inverseS(px, py) {
  if (!world) return [0, 0];
  if (rotateMap) {
      const y = world.H - (px - offsetX) / scale;
      const x = (py - offsetY) / scale;
      return [x, y];
  } else {
      const x = (px - offsetX) / scale;
      const y = (py - offsetY) / scale;
      return [x, y];
  }
}

function setStatus(txt, ok) {
  const badge = document.getElementById("connBadge");
  const dot   = document.getElementById("statusDot");
  if (badge) badge.textContent = ok ? "ACTIVE" : txt.toUpperCase();
  if (dot) dot.style.background = ok ? "#4a9e6a" : "#c0392b";
}

function speciesLabel(s) {
  s = (s || "").toLowerCase();
  if (s === "elephant")   return "Elephant";
  if (s === "leopard")    return "Leopard";
  if (s === "sloth_bear") return "Sloth Bear";
  if (s === "tiger")      return "Tiger";
  if (s === "lion")       return "Lion";
  if (s === "gaur")       return "Gaur";
  if (s === "spotted_deer") return "Spotted Deer";
  if (s === "sambar_deer") return "Sambar Deer";
  if (s === "human")      return "Human";
  return (s || "").replaceAll("_", " ").replace(/\b\w/g, ch => ch.toUpperCase());
}

function updateMapMetadata(initMsg) {
  const region = initMsg.region || {};
  const g = initMsg.world?.graph || null;
  const bbox = g?.bbox || region.bbox || null;
  const centerLat = Number(region.center_lat ?? 12.8008);
  const centerLon = Number(region.center_lon ?? 77.5756);
  const coords = document.getElementById("hud-coords");
  if (coords) coords.textContent = `${centerLat.toFixed(4)} N / ${centerLon.toFixed(4)} E`;

  const mapHud = document.getElementById("mapHud");
  if (mapHud && bbox) {
    const area = Number(region.declared_area_sq_km || g?.declared_area_sq_km || 260.51);
    const nodes = Number(g?.node_count || 0);
    const edges = Number(g?.edge_count || 0);
    mapHud.innerHTML =
      `BNP bbox ${bbox.south.toFixed(4)}-${bbox.north.toFixed(4)} N / ${bbox.west.toFixed(4)}-${bbox.east.toFixed(4)} E` +
      `<br>${area.toFixed(2)} sq km / OSM graph ${nodes} nodes, ${edges} edges`;
  }
}

function speciesEmoji(s) {
  s = (s || "").toLowerCase();
  if (s === "elephant")   return "🐘";
  if (s === "leopard")    return "🐆";
  if (s === "sloth_bear") return "🐻";
  if (s === "tiger")      return "T";
  if (s === "lion")       return "L";
  if (s.includes("deer")) return "D";
  if (s === "gaur")       return "G";
  if (s === "human")      return "👤";
  return "·";
}

function speciesColor(s) {
  s = (s || "").toLowerCase();
  if (s === "elephant")   return T.elephant;
  if (s === "leopard")    return T.leopard;
  if (s === "sloth_bear") return T.bear;
  if (s === "tiger")      return "#9c3b24";
  if (s === "lion")       return "#8f6a2d";
  if (s.includes("deer")) return "#7a6d42";
  if (s === "gaur")       return "#4f4637";
  if (s === "human")      return T.human;
  return T.ink3;
}

function getZoneTheme(t) {
  t = (t || "").toLowerCase();
  if (t === "water")      return { stroke: "#5a8fa8", fill: "rgba(90,143,168,0.08)",  label: "Water Body"  };
  if (t === "crop")       return { stroke: "#7a9e5a", fill: "rgba(122,158,90,0.08)",  label: "Agriculture" };
  if (t === "settlement") return { stroke: "#b08060", fill: "rgba(176,128,96,0.08)",  label: "Settlement"  };
  return                         { stroke: T.ink4,    fill: "rgba(100,100,90,0.06)",  label: "Zone"        };
}

function safeBase64ToBytes(b64) {
  let s = (b64 || "").trim().replace(/-/g, "+").replace(/_/g, "/");
  const pad = s.length % 4;
  if (pad === 2) s += "==";
  else if (pad === 3) s += "=";
  const raw = atob(s);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return out;
}

function getLayerBytes() {
  if (!world || !world.layers_u8) return null;
  const pack = world.layers_u8;
  const key  = `${world.W}x${world.H}|${simTime}`; 
  if (cachedLayerKey === key && cachedLayerBytes) return cachedLayerBytes;
  let data = null;
  try { if (pack && pack.b64) data = safeBase64ToBytes(pack.b64); } catch (_) {}
  if (!data) return null;
  cachedLayerKey   = key;
  cachedLayerBytes = data;
  return data;
}

function drawBackgroundComposite() {
  if (!world) return;
  const { W, H } = world;
  const data = getLayerBytes();
  if (!data) return;

  function idx(y, x, layer) { return (y * W + x) * 7 + layer; }

  if (
    !bgCanvas ||
    bgCanvas.width  !== W ||
    bgCanvas.height !== H ||
    bgCanvas.dataset.key !== cachedLayerKey
  ) {
    bgCanvas             = document.createElement("canvas");
    bgCanvas.width       = W;
    bgCanvas.height      = H;
    bgCanvas.dataset.key = cachedLayerKey;
    const bCtx = bgCanvas.getContext("2d");

    bCtx.fillStyle = "#e2ddd4";
    bCtx.fillRect(0, 0, W, H);

    const img = bCtx.getImageData(0, 0, W, H);
    const pix = img.data;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const o = (y * W + x) * 4;
        const water = data[idx(y, x, 0)] / 255;
        const crop  = data[idx(y, x, 1)] / 255;
        const sett  = data[idx(y, x, 3)] / 255;
        const cover = data[idx(y, x, 5)] / 255;

        let r = pix[o], g = pix[o+1], b = pix[o+2];

        if (cover > 0.05) {
          const t = Math.min(1, cover * 1.1);
          r = r*(1-t) + 168*t; g = g*(1-t) + 188*t; b = b*(1-t) + 148*t;
        }
        if (crop > 0.05) {
          const t = Math.min(1, crop * 1.0);
          r = r*(1-t) + 196*t; g = g*(1-t) + 210*t; b = b*(1-t) + 148*t;
        }
        if (sett > 0.05) {
          const t = Math.min(1, sett * 1.0);
          r = r*(1-t) + 210*t; g = g*(1-t) + 184*t; b = b*(1-t) + 148*t;
        }
        if (water > 0.05) {
          const t = Math.min(1, water * 1.2);
          r = r*(1-t) + 142*t; g = g*(1-t) + 186*t; b = b*(1-t) + 210*t;
        }

        pix[o]   = Math.round(r); pix[o+1] = Math.round(g); pix[o+2] = Math.round(b); pix[o+3] = 255;
      }
    }
    bCtx.putImageData(img, 0, 0);
  }

  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";

  // Ensure background rotation matches overlay transforms.
  if (rotateMap) {
    ctx.setTransform(
      0, scale, -scale, 0,
      offsetX + scale * H, offsetY 
    );
    ctx.drawImage(bgCanvas, 0, 0);
  } else {
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);
    ctx.drawImage(bgCanvas, 0, 0);
  }
  ctx.restore();
}

function drawOsmGraph() {
  if (!graph || !graph.nodes || !graph.edges) return;
  const dpr = devicePixelRatio;

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(70,68,60,0.28)";
  ctx.lineWidth = 0.8 * dpr;
  for (const e of graph.edges) {
    const a = graph.nodes[e.u];
    const b = graph.nodes[e.v];
    if (!a || !b) continue;
    ctx.beginPath();
    ctx.moveTo(sx(a.x, a.y), sy(a.x, a.y));
    ctx.lineTo(sx(b.x, b.y), sy(b.x, b.y));
    ctx.stroke();
  }
  ctx.restore();
}

function drawWaterResources() {
  if (!graph && !waterResources.length) return;
  const dpr = devicePixelRatio;

  if (graph && graph.water_polylines) {
    ctx.save();
    ctx.strokeStyle = "rgba(45,116,150,0.72)";
    ctx.lineWidth = 1.8 * dpr;
    ctx.lineCap = "round";
    for (const line of graph.water_polylines) {
      const pts = line.points || [];
      if (pts.length < 2) continue;
      ctx.beginPath();
      ctx.moveTo(sx(pts[0].x, pts[0].y), sy(pts[0].x, pts[0].y));
      for (let i = 1; i < pts.length; i++) ctx.lineTo(sx(pts[i].x, pts[i].y), sy(pts[i].x, pts[i].y));
      ctx.stroke();
    }
    ctx.restore();
  }

  ctx.save();
  for (const p of waterResources) {
    const r = (p.kind === "recharge_pit" ? 3.2 : 4.2) * dpr;
    ctx.fillStyle = p.kind === "recharge_pit" ? "rgba(44,95,138,0.72)" : "rgba(45,116,150,0.82)";
    ctx.beginPath();
    ctx.arc(sx(p.cx, p.cy), sy(p.cx, p.cy), r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawGrid() {
  if (!world) return;
  const dpr  = devicePixelRatio;
  const step = 50;

  ctx.save();
  ctx.strokeStyle = "rgba(80,70,50,0.10)";
  ctx.lineWidth   = 1 * dpr;

  for (let x = 0; x <= world.W; x += step) {
    ctx.beginPath();
    ctx.moveTo(sx(x, 0), sy(x, 0));
    ctx.lineTo(sx(x, world.H), sy(x, world.H));
    ctx.stroke();
  }
  for (let y = 0; y <= world.H; y += step) {
    ctx.beginPath();
    ctx.moveTo(sx(0, y), sy(0, y));
    ctx.lineTo(sx(world.W, y), sy(world.W, y));
    ctx.stroke();
  }
  ctx.restore();
}

function drawBorder() {
  if (!world) return;
  const dpr = devicePixelRatio;
  ctx.save();
  ctx.strokeStyle = "rgba(44,95,138,0.55)";
  ctx.lineWidth   = 1.5 * dpr;
  ctx.beginPath();
  ctx.moveTo(sx(0, 0), sy(0, 0));
  ctx.lineTo(sx(world.W, 0), sy(world.W, 0));
  ctx.lineTo(sx(world.W, world.H), sy(world.W, world.H));
  ctx.lineTo(sx(0, world.H), sy(0, world.H));
  ctx.closePath();
  ctx.stroke();
  ctx.strokeStyle = "rgba(0,0,0,0.10)";
  ctx.lineWidth   = 3 * dpr;
  ctx.beginPath();
  ctx.moveTo(sx(0, 0) - 2, sy(0, 0) - 2);
  ctx.lineTo(sx(world.W, 0) + 2, sy(world.W, 0) - 2);
  ctx.lineTo(sx(world.W, world.H) + 2, sy(world.W, world.H) + 2);
  ctx.lineTo(sx(0, world.H) - 2, sy(0, world.H) + 2);
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

// ─── VECTOR FENCE AND BRIDGE RENDERER ───
function drawFence() {
  if (!world) return;
  const dpr = devicePixelRatio;
  const fy = sy(0, world.H / 2);      // screen Y for y=H/2 line
  const cx = sx(world.W / 2, world.H / 2); // screen X at mid point
  const gap = 15 * scale;          // Width of the bridge gap

  ctx.save();
  
  // 1. Draw the Wooden Bridge in the gap
  ctx.fillStyle = "#8B5A2B"; // Sienna Wood color
  ctx.fillRect(cx - gap, fy - 6 * dpr, gap * 2, 12 * dpr);
  
  ctx.strokeStyle = "#5C3A21"; 
  ctx.lineWidth = 2 * dpr;
  ctx.strokeRect(cx - gap, fy - 6 * dpr, gap * 2, 12 * dpr);
  
  ctx.lineWidth = 1 * dpr;
  ctx.beginPath();
  for(let px = cx - gap + 4*dpr; px < cx + gap; px += 4*dpr) {
      ctx.moveTo(px, fy - 6*dpr);
      ctx.lineTo(px, fy + 6*dpr);
  }
  ctx.stroke();

  // 2. Draw the Fence lines
  ctx.strokeStyle = "#3c3732";
  ctx.lineWidth = 2.5 * dpr;
  
  ctx.beginPath();
  ctx.moveTo(sx(0, world.H / 2), fy);
  ctx.lineTo(cx - gap, fy);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(cx + gap, fy);
  ctx.lineTo(sx(world.W, world.H / 2), fy);
  ctx.stroke();
  
  // 3. Draw Fence Posts for realism
  ctx.fillStyle = "#221f1c";
  for (let x = 5; x <= world.W / 2 - 15; x += 10) {
     ctx.fillRect(sx(x, world.H / 2) - 1.5*dpr, fy - 3*dpr, 3*dpr, 6*dpr);
  }
  for (let x = world.W / 2 + 15; x < world.W; x += 10) {
     ctx.fillRect(sx(x, world.H / 2) - 1.5*dpr, fy - 3*dpr, 3*dpr, 6*dpr);
  }

  // 4. Add Label
  ctx.font = `600 ${8 * dpr}px 'Source Code Pro', monospace`;
  ctx.fillStyle = "#3c3732";
  ctx.textAlign = "center";
  ctx.fillText("ANIMAL CORRIDOR", cx, fy - 12 * dpr);

  ctx.restore();
}

function drawZones() {
  if (!zones || !showZones) return;
  const dpr = devicePixelRatio;

  for (const z of zones) {
    const theme = getZoneTheme(z.type);
    const x = sx(z.cx, z.cy), y = sy(z.cx, z.cy), r = z.radius * scale;

    ctx.save();
    ctx.fillStyle = theme.fill;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = theme.stroke;
    ctx.lineWidth   = 1 * dpr;
    ctx.setLineDash([4 * dpr, 5 * dpr]);
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = theme.stroke;
    ctx.beginPath(); ctx.arc(x, y, 2.5 * dpr, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }
}

function drawZoneLabels() {
  if (!zones || !showLabels) return;
  const dpr = devicePixelRatio;

  for (const z of zones) {
    const theme = getZoneTheme(z.type);
    const x = sx(z.cx, z.cy);
    const y = sy(z.cx, z.cy) - z.radius * scale - 12 * dpr;

    ctx.save();
    ctx.font         = `500 ${9.5 * dpr}px 'Source Sans 3', sans-serif`;
    ctx.textAlign    = "center";
    ctx.textBaseline = "middle";
    ctx.letterSpacing = "0.05em";

    const tw = ctx.measureText(theme.label).width;
    const pw = tw + 14 * dpr, ph = 15 * dpr;

    ctx.fillStyle   = "rgba(247,245,240,0.88)";
    ctx.strokeStyle = theme.stroke;
    ctx.lineWidth   = 1 * dpr;
    roundRect(ctx, x - pw/2, y - ph/2, pw, ph, 1.5 * dpr);
    ctx.fill(); ctx.stroke();

    ctx.fillStyle = T.ink2;
    ctx.fillText(theme.label, x, y);
    ctx.restore();
  }
}

function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y); c.lineTo(x + w - r, y);
  c.quadraticCurveTo(x + w, y, x + w, y + r);
  c.lineTo(x + w, y + h - r);
  c.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  c.lineTo(x + r, y + h);
  c.quadraticCurveTo(x, y + h, x, y + h - r);
  c.lineTo(x, y + r);
  c.quadraticCurveTo(x, y, x + r, y);
  c.closePath();
}

function drawHotspots() {
  if (!hotspots) return;
  const dpr = devicePixelRatio;
  hoveredHotspot = null;

  for (const h of hotspots) {
    if (h.type !== "mortality") continue;

    const x = sx(h.cx, h.cy), y = sy(h.cx, h.cy);
    const size = 5 * dpr;

    ctx.save();
    ctx.strokeStyle = "#c0392b";
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    ctx.moveTo(x - size, y - size); ctx.lineTo(x + size, y + size);
    ctx.moveTo(x + size, y - size); ctx.lineTo(x - size, y + size);
    ctx.stroke();

    const distSq = (mouseX * dpr - x)**2 + (mouseY * dpr - y)**2;
    if (distSq < (15 * dpr)**2) {
        hoveredHotspot = h;
        ctx.beginPath();
        ctx.arc(x, y, 10 * dpr, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(192, 57, 43, 0.5)";
        ctx.stroke();
    }
    ctx.restore();
  }
}

// ─── MODIFIED: DRAW AGENTS & GHOST TRAILS ───
function drawAgents() {
  hoveredAgent = null;
  const dpr = devicePixelRatio;

  for (const a of agents) {
    if (!a.alive) continue;
    const x     = sx(a.x, a.y), y = sy(a.x, a.y);
    const color = speciesColor(a.species);

    if (showTrails) {
      if (!trails.has(a.id)) trails.set(a.id, []);
      const arr = trails.get(a.id);
      arr.push({ x: a.x, y: a.y });
      while (arr.length > MAX_TRAIL) arr.shift();

      if (arr.length > 2) {
        const isSelected = selectedAgentId === a.id;
        const isGhosting = selectedAgentId !== null && !isSelected;

        if (!isGhosting || isSelected) {
          ctx.save();
          ctx.lineCap  = "round";
          ctx.lineJoin = "round";
          
          for (let i = 1; i < arr.length; i++) {
            const f = i / arr.length;
            ctx.globalAlpha = isSelected ? (f * 0.9) : (f * 0.30);
            
            if (isSelected) {
              ctx.strokeStyle = "#00ffff"; // Neon cyan for Forensic Mode
              ctx.lineWidth   = f * 3 * dpr;
              
              // Dynamic Danger Check (Dashed line if near crops/settlements)
              let inDanger = false;
              for(const z of zones) {
                if(z.type === 'settlement' || z.type === 'crop') {
                  if(Math.hypot(arr[i].x - z.cx, arr[i].y - z.cy) < z.radius) {
                    inDanger = true; break;
                  }
                }
              }
              if (inDanger) ctx.setLineDash([6 * dpr, 6 * dpr]);
              else ctx.setLineDash([]);
            } else {
              ctx.strokeStyle = color;
              ctx.lineWidth   = f * 1.5 * dpr;
            }

            ctx.beginPath();
            ctx.moveTo(sx(arr[i-1].x, arr[i-1].y), sy(arr[i-1].x, arr[i-1].y));
            ctx.lineTo(sx(arr[i].x, arr[i].y), sy(arr[i].x, arr[i].y));
            ctx.stroke();
          }
          ctx.globalAlpha = 1;
          ctx.restore();
        }
      }
    }

    const r   = 9 * dpr;
    const dxm = mouseX * dpr - x, dym = mouseY * dpr - y;
    const isH = dxm*dxm + dym*dym < (r + 5*dpr) * (r + 5*dpr);
    if (isH) hoveredAgent = a;

    ctx.save();
    
    // Dim unselected agents during forensic focus
    if (selectedAgentId !== null && a.id !== selectedAgentId) {
      ctx.globalAlpha = 0.2;
    } else {
      ctx.globalAlpha = 0.82;
    }

    ctx.fillStyle   = color;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = 1;

    ctx.strokeStyle = isH ? T.accent : "rgba(255,255,255,0.70)";
    ctx.lineWidth   = isH ? 1.5 * dpr : 1 * dpr;
    if (selectedAgentId === a.id) {
        ctx.strokeStyle = "#00ffff"; // Ring the selected agent
        ctx.lineWidth = 2.5 * dpr;
    }
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.stroke();

    if (isH) {
      ctx.strokeStyle = color;
      ctx.lineWidth   = 1 * dpr;
      ctx.globalAlpha = 0.4;
      ctx.beginPath(); ctx.arc(x, y, r * 1.7, 0, Math.PI * 2); ctx.stroke();
      ctx.globalAlpha = 1;
    }

    ctx.font         = `${10 * dpr}px ui-sans-serif`;
    ctx.textAlign    = "center";
    ctx.textBaseline = "middle";
    
    // Also dim the emoji if not selected
    if (selectedAgentId !== null && a.id !== selectedAgentId) ctx.globalAlpha = 0.2;
    ctx.fillText(speciesEmoji(a.species), x, y);
    ctx.globalAlpha = 1;

    if (showIntent) {
      const modeShort = { MIGRATE: "MIG", WATER: "H₂O", FOOD: "FRG", FLEE: "FLE", HUNT: "HNT", DEFEND: "DEF", RETURN_HOME: "RET", IDLE: ""}[a.mode] || "";
      if (modeShort) {
        ctx.font         = `600 ${7 * dpr}px 'Source Code Pro', monospace`;
        ctx.textAlign    = "left"; ctx.textBaseline = "middle";
        const tw  = ctx.measureText(modeShort).width + 5 * dpr;
        const tx  = x + r + 2 * dpr;
        const ty  = y - r + 1 * dpr;
        ctx.fillStyle = "rgba(247,245,240,0.85)";
        roundRect(ctx, tx - 1*dpr, ty - 4.5*dpr, tw, 10*dpr, 1.5*dpr);
        ctx.fill();
        ctx.fillStyle = T.ink3;
        ctx.fillText(modeShort, tx + 1.5*dpr, ty);
      }
      const bw  = 18 * dpr, bh = 2.5 * dpr, bx  = x - bw / 2, by  = y + r + 5 * dpr;
      const epct = Math.max(0, Math.min(1, a.energy / 100)), hpct = Math.max(0, Math.min(1, (100 - a.thirst) / 100));
      ctx.fillStyle = "rgba(255,255,255,0.55)";
      roundRect(ctx, bx, by, bw, bh, 1); ctx.fill();
      ctx.fillStyle = a.energy > 40 ? T.ok : T.warn;
      roundRect(ctx, bx, by, bw * epct, bh, 1); ctx.fill();

      ctx.fillStyle = "rgba(255,255,255,0.55)";
      roundRect(ctx, bx, by + bh + 2*dpr, bw, bh, 1); ctx.fill();
      ctx.fillStyle = T.accent;
      roundRect(ctx, bx, by + bh + 2*dpr, bw * hpct, bh, 1); ctx.fill();
    }
    ctx.restore();
  }
}

// ─── NEW: TELEMETRIC PROBE RENDERER ───
function drawProbe() {
  if (!showProbe || !probeTargetGrid) return;
  const dpr = devicePixelRatio;
  const tx = sx(probeTargetGrid.x, probeTargetGrid.y);
  const ty = sy(probeTargetGrid.x, probeTargetGrid.y);

  ctx.save();
  // Draw glowing target marker
  ctx.beginPath();
  ctx.arc(tx, ty, 8 * dpr, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0, 255, 255, 0.2)";
  ctx.fill();
  ctx.lineWidth = 2 * dpr;
  ctx.strokeStyle = "#00ffff";
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(tx, ty, 2 * dpr, 0, Math.PI * 2);
  ctx.fillStyle = "#00ffff";
  ctx.fill();

  // Draw filtered paths
  ctx.lineWidth = 1 * dpr;
  ctx.setLineDash([4 * dpr, 4 * dpr]);
  
  for (const a of agents) {
      if (!a.alive) continue;
      
      let match = false;
      if (probeFilter === "all") match = true;
      else if (probeFilter === "thirsty" && a.thirst > 70.0) match = true;
      else if (probeFilter === "starving" && a.energy < 30.0) match = true;
      else if (probeFilter === "stressed" && (a.mode === "DEFEND" || a.mode === "FLEE" || a.mode === "HUNT")) match = true;

      if (match) {
          const ax = sx(a.x, a.y);
          const ay = sy(a.x, a.y);
          
          // Draw gradient vector line
          const grad = ctx.createLinearGradient(ax, ay, tx, ty);
          grad.addColorStop(0, speciesColor(a.species));
          grad.addColorStop(1, "rgba(0, 255, 255, 0.8)");
          
          ctx.strokeStyle = grad;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          
          // Add an organic curve to the projection line
          const cx = (ax + tx) / 2 + (ty - ay) * 0.1;
          const cy = (ay + ty) / 2 + (ax - tx) * 0.1;
          ctx.quadraticCurveTo(cx, cy, tx, ty);
          ctx.stroke();
      }
  }
  ctx.restore();
}

function updateTooltip() {
  if (hoveredAgent) {
      const a   = hoveredAgent;
      const col = speciesColor(a.species);

      let condition = "Normal";
      if (a.thirst > 70.0) condition = "Severe Thirst";
      else if (a.energy < 30.0) condition = "Starving";
      else if (a.mode === "DEFEND") condition = "Stressed/Defending";
      else if (a.mode === "HUNT") condition = "Aggressive";
      else if (a.mode === "MIGRATE") condition = "Migrating";
      
      const modeLabel = { MIGRATE: "Migrating", WATER: "Seeking water", FOOD: "Foraging", FLEE: "Fleeing", HUNT: "Hunting", DEFEND: "Defending", RETURN_HOME: "Returning", IDLE: "Idle" }[a.mode] || a.mode;
      const weightStr = a.weight ? a.weight.toFixed(1) + " kg" : "N/A";
      const energy = a.energy.toFixed(1);
      const hyd    = (100 - a.thirst).toFixed(1);

      document.getElementById("tip-name").innerHTML =
        `<span>${speciesEmoji(a.species)}</span>` +
        `<span style="color:${col};font-weight:600">${speciesLabel(a.species)}</span>` +
        `<span style="font-size:.65rem;color:${T.ink4};margin-left:auto;font-family:'Source Code Pro',monospace">ID ${a.id}</span>`;

      document.getElementById("tip-rows").innerHTML = `
        <div class="tip-row"><span class="tip-key">Condition</span><span class="tip-val" style="color:${condition === 'Normal' ? T.ok : T.warn}">${condition}</span></div>
        <div class="tip-row"><span class="tip-key">Behaviour</span><span class="tip-val">${modeLabel}</span></div>
        <div class="tip-row"><span class="tip-key">Est. Mass</span><span class="tip-val">${weightStr}</span></div>
        <div class="tip-row" style="margin-top:4px;"><span class="tip-key">Energy</span><span class="tip-val">${energy}%</span></div>
        <div style="height:3px;background:${T.ruleLight};border-radius:1px;margin:2px 0 5px">
          <div style="height:100%;width:${energy}%;background:${a.energy>40?T.ok:T.warn};border-radius:1px;transition:width .3s"></div>
        </div>
        <div class="tip-row"><span class="tip-key">Hydration</span><span class="tip-val">${hyd}%</span></div>
        <div style="height:3px;background:${T.ruleLight};border-radius:1px;margin:2px 0 0">
          <div style="height:100%;width:${hyd}%;background:${T.accent};border-radius:1px;transition:width .3s"></div>
        </div>
      `;
      positionTooltip();
  } 
  else if (hoveredHotspot) {
      document.getElementById("tip-name").innerHTML = `<span>❌</span><span style="color:#c0392b;font-weight:600">Mortality Event</span>`;
      document.getElementById("tip-rows").innerHTML = `<div class="tip-row"><span class="tip-key">Cause</span><span class="tip-val" style="color:${T.ink}">${hoveredHotspot.label}</span></div>`;
      positionTooltip();
  } else {
      tooltip.style.display = "none";
  }
}

function positionTooltip() {
  const cw   = document.getElementById("cw");
  const rect = cw.getBoundingClientRect();
  let lx = mouseX + 16, ly = mouseY + 12;
  if (lx + 185 > rect.width)  lx = mouseX - 200;
  if (ly + 175 > rect.height) ly = mouseY - 185;
  tooltip.style.left    = lx + "px";
  tooltip.style.top     = ly + "px";
  tooltip.style.display = "block";
}

function draw() {
  if (!canvas || !ctx) return;
  applyViewMode();
  if (show3D && window.WildSim3D && window.WildSim3D.isReady && window.WildSim3D.isReady()) {
    requestAnimationFrame(draw);
    return;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = T.paper2;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  drawBackgroundComposite();
  drawOsmGraph();
  drawWaterResources();
  drawGrid();
  drawBorder();
  if (!graph) drawFence();
  drawZones();

  const cellW = world.W / GRID_W;
  const cellH = world.H / GRID_H;
  drawMapLandmarks(ctx, cellW, cellH);

  drawAgents();
  drawProbe(); // <--- INJECTED FORENSIC PROBE
  drawHotspots();
  drawZoneLabels();
  updateTooltip();
  frameCount++;
  requestAnimationFrame(draw);
}

function addLog(text, type) {
  const list = document.getElementById("log-list");
  if (!list) return;
  const el = document.createElement("div");
  el.className = "log-entry";
  const ts = String(simTime).padStart(5, "0");
  el.innerHTML = `<span style="color:${T.ink4}">t=${ts}</span>  ${text}`;
  list.prepend(el);
  while (list.children.length > 28) list.removeChild(list.lastChild);
}

function updateSidebar(msg) {
  const alive = agents.filter(a => a.alive);
  const el  = alive.filter(a => a.species === "elephant"  ).length;
  const lp  = alive.filter(a => a.species === "leopard"   ).length;
  const tg  = alive.filter(a => a.species === "tiger"     ).length;
  const li  = alive.filter(a => a.species === "lion"      ).length;
  const sb  = alive.filter(a => a.species === "sloth_bear").length;
  const hu  = alive.filter(a => a.species === "human"     ).length;
  const tot = alive.length;
  const speciesN = new Set(alive.map(a => a.species)).size;
  const mx  = Math.max(el, lp, tg, li, sb, hu, 1);

  set("pop-total", tot);
  set("pop-step",  `STEP ${msg.t} · ${speciesN} SPECIES`);
  set("hud-step",  String(msg.t).padStart(5, "0"));
  set("fp",        `N = ${tot}`);
  set("fs",        `t = ${msg.t}`);

  set("cnt-el", el); setBarW("bar-el", el / mx * 100);
  set("cnt-lp", lp); setBarW("bar-lp", lp / mx * 100);
  set("cnt-tg", tg); setBarW("bar-tg", tg / mx * 100);
  set("cnt-li", li); setBarW("bar-li", li / mx * 100);
  set("cnt-sb", sb); setBarW("bar-sb", sb / mx * 100);
  set("cnt-hu", hu); setBarW("bar-hu", hu / mx * 100);

  if (msg.analytics) {
    const cap = Math.min(100, msg.analytics.carrying_capacity || 0);
    setBarW("carry-fill", cap);
    set("carry-pct", cap.toFixed(0) + "%");

    let bal = 50;
    const scores = msg.analytics.eco_scores;
    if (scores) {
      const vals = Object.values(scores);
      if (vals.length) bal = vals.reduce((a, b) => a + b, 0) / vals.length;
    }
    set("eco-bal", bal.toFixed(0) + "%");
    
    if (msg.analytics.intents) {
      const intentBox = document.getElementById("intent-stats");
      if (intentBox) {
        let intentHtml = "";
        for (const [intentName, count] of Object.entries(msg.analytics.intents)) {
          const pct = tot > 0 ? Math.round((count / tot) * 100) : 0;
          intentHtml += `
            <div style="display: flex; flex-direction: column; gap: 2px;">
              <div style="display: flex; justify-content: space-between;">
                <span>${intentName}</span>
                <span style="font-weight: 600; color: var(--ink)">${pct}%</span>
              </div>
              <div style="height: 2px; background: var(--rule-light); border-radius: 1px;">
                <div style="height: 100%; width: ${pct}%; background: var(--accent); border-radius: 1px;"></div>
              </div>
            </div>
          `;
        }
        intentBox.innerHTML = intentHtml || '<span style="color: var(--ink-4)">No active intent data...</span>';
      }
    }

    if (msg.analytics.new_conflicts && msg.analytics.new_conflicts.length > 0) {
      msg.analytics.new_conflicts.forEach(conflictMsg => {
        addLog(`<span style="color:var(--warn); font-weight: 600;">[HWC ALERT]</span> ${conflictMsg}`, "conflict");
      });
    }
  }

  let wz = 0, crz = 0, stz = 0;
  for (const a of alive) {
    for (const z of zones) {
      if (Math.hypot(a.x - z.cx, a.y - z.cy) <= z.radius) {
        if      (z.type === "water")      wz++;
        else if (z.type === "crop")       crz++;
        else if (z.type === "settlement") stz++;
        break;
      }
    }
  }

  set("zc-water",  wz);
  set("za-water",  wz  ? `${wz} agent${wz  > 1 ? "s" : ""} present` : "Unoccupied");
  set("zc-crop",   crz);
  set("za-crop",   crz ? `${crz} foraging`      : "No activity");
  set("zc-settle", stz);
  set("za-settle", stz ? `${stz} in proximity`  : "Unoccupied");

  set("eco-conflict", msg.hotspots ? msg.hotspots.length : 0);
  set("eco-range", Math.min(100, Math.max(0, 40 + tot * 1.5)).toFixed(0) + "%");
}

function set(id, val)     { const el = document.getElementById(id); if (el) el.textContent = val; }
function setBarW(id, pct) { const el = document.getElementById(id); if (el) el.style.width  = pct + "%"; }

// ─── NEW: FORENSIC CLICK INTERACTION ───
const cw = document.getElementById("cw");
if (cw) {
  cw.addEventListener("mousemove", e => {
    const r = cw.getBoundingClientRect();
    mouseX = e.clientX - r.left;
    mouseY = e.clientY - r.top;
  });
  
  cw.addEventListener("mousedown", e => {
    if (hoveredAgent) {
        // If they click an agent, highlight its Ghost Trail
        if (showGhostTrails) {
            selectedAgentId = hoveredAgent.id === selectedAgentId ? null : hoveredAgent.id;
            probeTargetGrid = null; // Clear the probe
        }
    } else if (showProbe) {
        // If they click empty space with the Probe active, set the target
        const [gx, gy] = inverseS(mouseX, mouseY);
        probeTargetGrid = { x: gx, y: gy };
        selectedAgentId = null; // Clear agent selection
    } else {
        selectedAgentId = null;
        probeTargetGrid = null;
    }
  });

  cw.addEventListener("mouseleave", () => {
    hoveredAgent = null;
    hoveredHotspot = null;
    if (tooltip) tooltip.style.display = "none";
  });
}

async function loadTopologies() {
  const res = await fetch("/api/topologies");
  const js  = await res.json();
  const sel = document.getElementById("topology");
  sel.innerHTML = "";
  for (const t of js.topologies) {
    const opt = document.createElement("option");
    opt.value = t.id; opt.textContent = t.name;
    sel.appendChild(opt);
  }
}

async function loadConfig() {
  const res = await fetch("/api/config");
  const js  = await res.json();
  document.getElementById("topology").value = js.defaults.topology;
  document.getElementById("seed").value     = js.defaults.seed;
  document.getElementById("fps").value      = js.defaults.fps;
  document.getElementById("useBaseline").checked = !!js.defaults.use_bannerghatta_baseline;
  const c = js.defaults.species_counts || {};
  if (c.elephant !== undefined) document.getElementById("c_el").value = c.elephant;
  if (c.leopard !== undefined) document.getElementById("c_lp").value = c.leopard;
  if (c.sloth_bear !== undefined) document.getElementById("c_sb").value = c.sloth_bear;
  if (c.tiger !== undefined) document.getElementById("c_tg").value = c.tiger;
  if (c.lion !== undefined) document.getElementById("c_li").value = c.lion;
  if (c.human !== undefined) document.getElementById("c_hu").value = c.human;
}

function bindUI() {
  const toggles = {
    showZones:  v => showZones  = v,
    showIntent: v => showIntent = v,
    showTrails: v => showTrails = v,
    showLabels: v => showLabels = v,
    show3D:     v => { show3D = v; applyViewMode(); },
  };
  for (const [id, fn] of Object.entries(toggles)) {
    document.getElementById(id)?.addEventListener("change", e => fn(e.target.checked));
  }
  
  // NEW: Bind Forensic Controls
  document.getElementById("showGhostTrails")?.addEventListener("change", e => showGhostTrails = e.target.checked);
  document.getElementById("showProbe")?.addEventListener("change", e => {
      showProbe = e.target.checked;
      if (!showProbe) probeTargetGrid = null;
  });
  document.getElementById("probeFilter")?.addEventListener("change", e => probeFilter = e.target.value);

  document.getElementById("useBaseline")?.addEventListener("change", e => {
    for (const id of ["c_el", "c_lp", "c_sb", "c_tg", "c_li", "c_hu"]) {
      const input = document.getElementById(id);
      if (input) input.disabled = e.target.checked;
    }
  });
  document.getElementById("useBaseline")?.dispatchEvent(new Event("change"));

  document.getElementById("apply")?.addEventListener("click", async () => {
    const useBaseline = document.getElementById("useBaseline")?.checked;
    const payload = {
      topology: document.getElementById("topology").value,
      seed:     parseInt(document.getElementById("seed").value || "1",  10),
      fps:      parseInt(document.getElementById("fps").value  || "20", 10),
      use_bannerghatta_baseline: !!useBaseline,
    };
    if (!useBaseline) {
      payload.species_counts = {
          elephant:   parseInt(document.getElementById("c_el").value || "0", 10),
          leopard:    parseInt(document.getElementById("c_lp").value || "0", 10),
          sloth_bear: parseInt(document.getElementById("c_sb").value || "0", 10),
          tiger:      parseInt(document.getElementById("c_tg").value || "0", 10),
          lion:       parseInt(document.getElementById("c_li").value || "0", 10),
          human:      parseInt(document.getElementById("c_hu").value || "0", 10),
      };
    }
    pendingReset = true;
    trails.clear();
    prevAgentStates.clear();
    selectedAgentId = null;
    probeTargetGrid = null;
    setStatus("Resetting...", false);
    document.getElementById("log-list").innerHTML = "";
    await fetch("/api/reset", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
  });
}

function connectWS() {
  if (ws) { try { ws.close(); } catch (_) {} ws = null; }
  const proto = location.protocol === "https:" ? "wss://" : "ws://";
  ws = new WebSocket(proto + location.host + "/ws");

  ws.onopen  = () => setStatus("Connected", true);
  ws.onclose = () => { setStatus("Offline", false); setTimeout(connectWS, 900); };
  ws.onerror = () => setStatus("Error", false);

  ws.onmessage = ev => {
    const msg = JSON.parse(ev.data);

    if (msg.type === "init") {
      window.__wildsimLatestInit = msg;
      topo  = msg.topology;
      world = msg.world;
      graph = msg.world?.graph || null;
      waterResources = msg.water_resources || [];
      if (topo?.id) document.getElementById("topology").value = topo.id;
      zones = msg.zones || [];
      updateMapMetadata(msg);
      const stageEl = document.getElementById("stageTitle");
      if (stageEl) stageEl.textContent = `TOPOLOGY: ${(topo?.name || "").toUpperCase()}`;
      cachedLayerBytes = null;
      cachedLayerKey   = "";
      bgCanvas         = null;
      pendingReset     = false;
      setStatus("Active", true);
      fitCanvas();
      applyViewMode();
      requestAnimationFrame(draw);
    }

    if (msg.type === "state") {
      if (pendingReset) return;
      simTime = msg.t;

      const newAgents = msg.agents || [];
      if (msg.zones)    zones    = msg.zones;
      if (msg.hotspots) hotspots = msg.hotspots;
      if (msg.water_basins) waterResources = msg.water_basins;

      for (const a of newAgents) {
        const prev = prevAgentStates.get(a.id);
        if (prev) {
          if (prev.alive && !a.alive) {
            addLog(`<span style="color:${T.leopard}">${speciesLabel(a.species)} (ID ${a.id}) perished</span>`, "death");
          } else if (prev.mode !== "MIGRATE" && a.mode === "MIGRATE") {
            addLog(`${speciesLabel(a.species)} (ID ${a.id}) — initiated migration`);
          } else if (prev.mode !== "DEFEND" && a.mode === "DEFEND" && Math.random() < 0.3) {
            addLog(`${speciesLabel(a.species)} (ID ${a.id}) — inter-species conflict`);
          } else if (prev.mode !== "WATER" && a.mode === "WATER" && Math.random() < 0.06) {
            addLog(`${speciesLabel(a.species)} (ID ${a.id}) — reached water source`);
          }
        }
        prevAgentStates.set(a.id, { alive: a.alive, mode: a.mode });
      }

      agents = newAgents;
      window.__wildsimLatestState = msg;
      updateSidebar(msg);
    }
  };
}

(async function main() {
  await loadTopologies();
  await loadConfig();
  bindUI();
  fitCanvas();
  applyViewMode();
  connectWS();
  requestAnimationFrame(draw);
})();