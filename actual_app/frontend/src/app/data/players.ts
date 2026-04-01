export type Position = 'QB' | 'RB' | 'WR' | 'TE' | 'OL' | 'DL' | 'LB' | 'DB';
export type TierLabel = 'High Upside' | 'Safe Floor' | 'Boom or Bust' | 'Developmental' | 'Overdrafted';

export interface Player {
  id: string;
  name: string;
  position: Position;
  college: string;
  draftRound: number;
  draftPick: number;
  projectedPick: number;
  sleeperScore: number;
  surplusValue: number; // draftPick - projectedPick (positive = undervalued)
  breakoutProbability: number;
  availabilityFactor: number;
  tier: TierLabel;
  height: number; // total inches
  weight: number; // lbs
  combine: {
    fortyYard?: number;
    vertical?: number;
    broadJump?: number;
    threeCone?: number;
    shuttle?: number;
    benchPress?: number;
  };
  athleticRadar: {
    speed: number;
    burst: number;
    agility: number;
    power: number;
    frame: number;
    stature: number;
  };
  featureImportance: {
    draftCapital: number;
    agility: number;
    collegeProduction: number;
    athleticism: number;
    size: number;
  };
}

export const modelConfidence: Record<Position, string> = {
  QB: 'Correctly ranks about 6 in 10 QBs by career trajectory',
  RB: 'Correctly ranks about 7 in 10 RBs by career trajectory',
  WR: 'Correctly ranks about 7 in 10 WRs by career trajectory',
  TE: 'Correctly ranks about 6 in 10 TEs by career trajectory',
  OL: 'Correctly ranks about 5 in 10 OLs by career trajectory',
  DL: 'Correctly ranks about 6 in 10 DLs by career trajectory',
  LB: 'Correctly ranks about 6 in 10 LBs by career trajectory',
  DB: 'Correctly ranks about 7 in 10 DBs by career trajectory',
};

export const positionAvgRadar: Record<string, Record<string, number>> = {
  WR: { speed: 54, burst: 60, agility: 57, power: 42, frame: 40, stature: 40 },
  TE: { speed: 56, burst: 47, agility: 41, power: 60, frame: 70, stature: 65 },
  QB: { speed: 33, burst: 36, agility: 43, power: 64, frame: 48, stature: 56 },
  RB: { speed: 50, burst: 51, agility: 50, power: 50, frame: 49, stature: 49 },
};

// Map sleeper_score and surplus_value to a tier
function getTier(score: number, surplus: number): TierLabel {
  if (score > 75) return 'High Upside';
  if (score > 55) {
    return surplus > 5 ? 'Safe Floor' : 'Boom or Bust';
  }
  if (score > 40) return 'Developmental';
  return 'Overdrafted';
}

function toTitleCase(str: string): string {
  if (!str) return '';
  return str.toLowerCase().replace(/(^|[^\w])(\w)/g, (match, p1, p2) => p1 + p2.toUpperCase());
}

export async function fetchPlayers(): Promise<Player[]> {
  try {
    const res = await fetch('http://localhost:8000/prospects?limit=500');
    if (!res.ok) throw new Error('API fetch failed');
    const data = await res.json();

    return data.map((d: any) => {
      const draftPick = d.draft_pick || 0;
      const surplus = Math.round(d.surplus_value || 0);
      const projectedPick = Math.max(1, draftPick - surplus);
      
      // Helper to scale Z-score (-3 to 3) to 0-100
      const scaleZ = (z: number | undefined) => {
        if (z === undefined) return 50;
        return Math.round(Math.max(0, Math.min(100, (z + 3) / 6 * 100)));
      };

      return {
        id: String(d.id || Math.random()),
        name: d.player_name || 'Unknown',
        position: (d.position || 'Unknown') as Position,
        college: toTitleCase(d.college || 'Unknown'),
        draftRound: d.draft_round || 0,
        draftPick: draftPick,
        projectedPick: projectedPick,
        sleeperScore: Math.round(d.sleeper_score || 0),
        surplusValue: surplus,
        breakoutProbability: Math.round((d.pro_bowl_probability || 0) * 100),
        availabilityFactor: Math.round((d.availability_factor || 0) * 100),
        tier: getTier(d.sleeper_score || 0, d.surplus_value || 0),
        height: d.height_in || 0,
        weight: d.weight_lbs || 0,
        combine: {
          fortyYard: d.forty_yard,
          vertical: d.vertical_jump,
          broadJump: d.broad_jump,
          threeCone: d.cone_drill,
          shuttle: d.shuttle,
          benchPress: d.bench_reps
        },
        athleticRadar: {
          speed: scaleZ(d.speed_score_z),
          burst: scaleZ(d.burst_score_z),
          agility: d.agility_score_z ? Math.round(Math.max(0, Math.min(100, (-d.agility_score_z + 3) / 6 * 100))) : 50,
          power: scaleZ(d.bench_reps_z),
          frame: scaleZ(d.bmi_z),
          stature: scaleZ(d.height_in_z)
        },
        featureImportance: {
          draftCapital: Math.round((d.surplus_value_percentile || 0.5) * 100),
          agility: d.agility_score_z ? Math.round(Math.max(0, Math.min(100, (-d.agility_score_z + 3) / 6 * 100))) : 50,
          collegeProduction: Math.round((d.peak_value_percentile || 0.5) * 100),
          athleticism: scaleZ(d.speed_score_z),
          size: scaleZ(d.bmi_z)
        }
      };
    });
  } catch (error) {
    console.error("Failed to fetch players:", error);
    return [];
  }
}

// Keep export utilities for components
export function getTierColors(tier: TierLabel) {
  switch (tier) {
    case 'High Upside': return { hex: '#39ff14', bg: 'bg-[rgba(57,255,20,0.1)]', bgHover: 'hover:bg-[rgba(57,255,20,0.2)]', bgSelected: 'bg-[rgba(57,255,20,0.2)]', text: 'text-[#39ff14]', border: 'border-[#39ff14]/30' };
    case 'Safe Floor': return { hex: '#60a5fa', bg: 'bg-blue-500/10', bgHover: 'hover:bg-blue-500/20', bgSelected: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-400/30' };
    case 'Boom or Bust': return { hex: '#c084fc', bg: 'bg-purple-500/10', bgHover: 'hover:bg-purple-500/20', bgSelected: 'bg-purple-500/20', text: 'text-purple-400', border: 'border-purple-400/30' };
    case 'Developmental': return { hex: '#facc15', bg: 'bg-yellow-500/10', bgHover: 'hover:bg-yellow-500/20', bgSelected: 'bg-yellow-500/20', text: 'text-yellow-400', border: 'border-transparent' };
    case 'Overdrafted': return { hex: '#f87171', bg: 'bg-red-500/10', bgHover: 'hover:bg-red-500/20', bgSelected: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-400/30' };
    default: return { hex: '#94a3b8', bg: 'bg-slate-500/10', bgHover: 'hover:bg-slate-500/20', bgSelected: 'bg-slate-500/20', text: 'text-slate-400', border: 'border-slate-400/30' };
  }
}

export function getScoreColor(score: number): string {
  if (score >= 80) return '#39ff14';
  if (score >= 65) return '#60a5fa'; // blue-400
  if (score >= 50) return '#facc15'; // yellow-400
  return '#f87171'; // red-400
}

export function getScoreClass(score: number): string {
  if (score >= 80) return 'text-[#39ff14]';
  if (score >= 65) return 'text-blue-400';
  if (score >= 50) return 'text-yellow-400';
  return 'text-red-400';
}

export function formatHeight(inches: number): string {
  if (!inches) return '-';
  const feet = Math.floor(inches / 12);
  const remainingInches = inches % 12;
  return `${feet}'${remainingInches}"`;
}

export function formatPick(round: number, pick: number): string {
  if (!round || !pick) return 'Undrafted';
  return `Rd ${round} Pk ${pick}`;
}