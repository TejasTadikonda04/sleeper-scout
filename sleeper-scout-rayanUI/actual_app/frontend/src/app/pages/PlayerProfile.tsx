import { useParams, useNavigate } from 'react-router';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Legend, Tooltip,
} from 'recharts';
import { ArrowLeft, Users, TrendingUp, Shield, AlertCircle, Award, User } from 'lucide-react';
import { useEffect, useState } from 'react';
import {
  fetchPlayers, Player, formatHeight, formatPick, getScoreColor, getScoreClass, getTierColors,
  modelConfidence, TierLabel, positionAvgRadar,
} from '../data/players';

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-card border border-border rounded-xl p-5">
      <h3 className="text-muted-foreground text-xs uppercase tracking-widest mb-4" style={{ fontWeight: 600 }}>{title}</h3>
      {children}
    </div>
  );
}

function StatPill({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-secondary rounded-lg px-4 py-3">
      <div className="text-muted-foreground text-xs" style={{ textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>{label}</div>
      <div className="text-foreground mt-0.5 text-lg" style={{ fontWeight: 700 }}>{value}</div>
      {sub && <div className="text-muted-foreground text-xs mt-0.5">{sub}</div>}
    </div>
  );
}

const featureLabels: Record<string, string> = {
  draftCapital: 'Draft Capital',
  agility: 'Agility',
  collegeProduction: 'College Production',
  athleticism: 'Athleticism',
  size: 'Size',
};

const featureColors: Record<string, string> = {
  draftCapital: 'var(--color-primary)',
  agility: '#8b5cf6',
  collegeProduction: 'var(--color-success)',
  athleticism: 'var(--color-warning)',
  size: '#ec4899',
};

export function PlayerProfile() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [players, setPlayers] = useState<Player[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPlayers().then(data => {
      setPlayers(data);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <div className="min-h-full bg-background flex items-center justify-center">
        <p className="text-muted-foreground animate-pulse">Loading player data...</p>
      </div>
    );
  }

  const player = players.find(p => p.id === id);

  if (!player) {
    return (
      <div className="min-h-full bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="text-muted-foreground text-6xl mb-4">404</div>
          <p className="text-muted-foreground">Player not found</p>
          <button
            onClick={() => navigate('/')}
            className="mt-4 text-success text-sm hover:text-success/80"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const { bg: tierBg, text: tierText, border: tierBorder } = getTierColors(player.tier);
  const scoreColor = getScoreColor(player.sleeperScore);

  const posAvg = positionAvgRadar[player.position] || { speed: 50, burst: 50, agility: 50, power: 50, frame: 50, stature: 50 };

  // Build radar data (skip nulls)
  const radarMetrics = [
    { metric: 'Speed', key: 'speed' },
    { metric: 'Burst', key: 'burst' },
    { metric: 'Agility', key: 'agility' },
    { metric: 'Power', key: 'power' },
    { metric: 'Frame', key: 'frame' },
    { metric: 'Stature', key: 'stature' },
  ];
  const radarData = radarMetrics
    .filter(m => player.athleticRadar[m.key as keyof typeof player.athleticRadar] !== null)
    .map(m => ({
      metric: m.metric,
      player: player.athleticRadar[m.key as keyof typeof player.athleticRadar] as number,
      posAvg: posAvg[m.key as keyof typeof posAvg],
    }));

  // Surplus value bar
  const surplusClamp = Math.max(-50, Math.min(50, player.surplusValue));
  const surplusPct = ((surplusClamp + 50) / 100) * 100;
  const isPositiveSurplus = player.surplusValue >= 0;

  // Availability color
  const availColor =
    player.availabilityFactor >= 75 ? 'var(--color-success)' :
    player.availabilityFactor >= 55 ? 'var(--color-warning)' :
    'var(--color-destructive)';

  // Sorted feature importance
  const sortedFeatures = Object.entries(player.featureImportance)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5);

  const confidence = modelConfidence[player.position];

  return (
    <div className="min-h-full bg-background p-6">
      {/* Back nav */}
      <div className="flex items-center gap-3 mb-5">
        <button
          onClick={() => navigate(-1)}
          className="flex items-center gap-1.5 text-muted-foreground hover:text-foreground text-sm transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        <span className="text-muted-foreground">/</span>
        <div className="flex items-center gap-2">
          <User className="w-4 h-4 text-primary" />
          <span className="text-foreground text-sm" style={{ fontWeight: 600 }}>{player.name}</span>
        </div>
      </div>

      {/* Hero Header */}
      <div className="bg-card border border-border rounded-xl p-6 mb-5">
        <div className="flex flex-wrap items-start gap-5">
          {/* Score circle */}
          <div
            className="w-20 h-20 rounded-2xl flex flex-col items-center justify-center flex-shrink-0"
            style={{ background: `${scoreColor}15`, border: `2px solid ${scoreColor}40` }}
          >
            <div className="text-3xl" style={{ color: scoreColor, fontWeight: 800, lineHeight: 1 }}>{player.sleeperScore}</div>
            <div className="text-xs mt-0.5" style={{ color: scoreColor, opacity: 0.7, fontWeight: 600 }}>SCORE</div>
          </div>

          {/* Player info */}
          <div className="flex-1">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h1 className="text-foreground text-2xl" style={{ fontWeight: 800 }}>{player.name}</h1>
                <div className="flex items-center gap-2 mt-1.5 flex-wrap">
                  <span className="bg-secondary text-foreground text-xs px-2 py-0.5 rounded" style={{ fontWeight: 700 }}>{player.position}</span>
                  <span className="text-muted-foreground text-sm">{player.college}</span>
                  <span className="text-muted-foreground">·</span>
                  <span className="text-muted-foreground text-sm">{formatPick(player.draftRound, player.draftPick)}</span>
                  <span className="text-muted-foreground">·</span>
                  <span className="text-muted-foreground text-sm">{formatHeight(player.height)} / {player.weight} lbs</span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className={`inline-flex items-center px-3 py-1 rounded-lg text-sm ${tierBg} ${tierText} border ${tierBorder}`} style={{ fontWeight: 700 }}>
                  {player.tier}
                </span>
                <button
                  onClick={() => navigate(`/compare?p1=${player.id}`)}
                  className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-secondary border border-border text-muted-foreground hover:text-foreground hover:border-muted-foreground transition-all text-sm"
                >
                  <Users className="w-3.5 h-3.5" />
                  Compare
                </button>
              </div>
            </div>

            {/* Key stats row */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
              <StatPill label="Sleeper Score" value={player.sleeperScore} sub="out of 100" />
              <StatPill
                label="Surplus Value"
                value={`${player.surplusValue >= 0 ? '+' : ''}${player.surplusValue}`}
                sub={player.surplusValue >= 0 ? 'Undervalued pick' : 'Overdrafted'}
              />
              <StatPill label="Breakout Prob." value={`${player.breakoutProbability}%`} sub="Career upside" />
              <StatPill label="Durability" value={`${player.availabilityFactor}%`} sub="Availability factor" />
            </div>
          </div>
        </div>
      </div>

      {/* Main grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

        {/* Surplus Value Bar */}
        <Section title="Surplus Value Gauge">
          <div className="space-y-4">
            {/* Labels */}
            <div className="flex justify-between text-xs text-muted-foreground font-mono">
              <span>−50 (Overdrafted)</span>
              <span>0</span>
              <span>+50 (Undervalued)</span>
            </div>

            {/* Bar */}
            <div className="relative h-8 rounded-lg overflow-hidden flex">
              {/* Red left half */}
              <div className="flex-1 h-full" style={{ background: 'linear-gradient(to right, #7f1d1d, var(--color-destructive))' }} />
              {/* Green right half */}
              <div className="flex-1 h-full" style={{ background: 'linear-gradient(to right, var(--color-success), #065f46)' }} />

              {/* Center line */}
              <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-card/60" />

              {/* Player dot */}
              <div
                className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full border-2 border-background shadow-lg z-10 transition-all"
                style={{
                  left: `calc(${surplusPct}% - 8px)`,
                  backgroundColor: isPositiveSurplus ? 'var(--color-success)' : 'var(--color-destructive)',
                  boxShadow: `0 0 12px ${isPositiveSurplus ? 'var(--color-success)' : 'var(--color-destructive)'}80`,
                }}
              />
            </div>

            {/* Caption */}
            <div className="bg-secondary/60 rounded-lg p-3 border border-border/50">
              <p className="text-foreground text-sm">
                {isPositiveSurplus ? (
                  <>Projected to produce like a <span className="text-success" style={{ fontWeight: 700 }}>Pick {player.projectedPick}</span> player — drafted at <span className="text-muted-foreground" style={{ fontWeight: 600 }}>Pick {player.draftPick}</span>. <span className="text-success">Strong value pick.</span></>
                ) : (
                  <>Projected to produce like a <span className="text-destructive" style={{ fontWeight: 700 }}>Pick {player.projectedPick}</span> player — drafted at <span className="text-muted-foreground" style={{ fontWeight: 600 }}>Pick {player.draftPick}</span>. <span className="text-destructive">High draft capital investment.</span></>
                )}
              </p>
            </div>
          </div>
        </Section>

        {/* Availability Meter */}
        <Section title="Durability Projection">
          <div className="space-y-4">
            <div className="flex items-end justify-between">
              <div>
                <div className="text-4xl" style={{ color: availColor, fontWeight: 800 }}>{player.availabilityFactor}%</div>
                <div className="text-muted-foreground text-sm mt-1">Availability Factor</div>
              </div>
              <div className="text-right">
                <div className="text-muted-foreground text-sm" style={{ fontWeight: 600 }}>
                  {player.availabilityFactor >= 80 ? 'High Durability' :
                   player.availabilityFactor >= 65 ? 'Moderate Durability' :
                   player.availabilityFactor >= 50 ? 'Some Injury Risk' :
                   'Significant Injury Risk'}
                </div>
                <div className="text-muted-foreground text-xs mt-0.5">Based on injury history + build</div>
              </div>
            </div>

            {/* Bar */}
            <div className="h-4 bg-secondary rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{
                  width: `${player.availabilityFactor}%`,
                  background: `linear-gradient(to right, var(--color-destructive), var(--color-warning) ${50}%, ${availColor})`,
                }}
              />
            </div>

            {/* Ticks */}
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>0%</span>
              <span>25%</span>
              <span>50%</span>
              <span>75%</span>
              <span>100%</span>
            </div>

            {/* Risk flags */}
            <div className="grid grid-cols-3 gap-2 pt-2">
              {[
                { label: 'Practice Time', val: Math.round(player.availabilityFactor * 0.95) },
                { label: 'Game Availability', val: Math.round(player.availabilityFactor * 0.88) },
                { label: '3-Yr Projection', val: Math.round(player.availabilityFactor * 0.78) },
              ].map(item => (
                <div key={item.label} className="bg-secondary rounded-lg p-2.5 text-center">
                  <div className="text-sm" style={{ color: availColor, fontWeight: 700 }}>{item.val}%</div>
                  <div className="text-muted-foreground text-xs mt-0.5">{item.label}</div>
                </div>
              ))}
            </div>
          </div>
        </Section>

        {/* Athletic Radar Chart */}
        <Section title="Athletic Profile — vs. Position Group Average">
          {radarData.length > 0 ? (
            <div>
              <ResponsiveContainer width="100%" height={280}>
                <RadarChart data={radarData} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
                  <PolarGrid stroke="var(--color-border)" />
                  <PolarAngleAxis
                    dataKey="metric"
                    tick={{ fill: 'var(--color-muted-foreground)', fontSize: 11, fontWeight: 600 }}
                  />
                  <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                  <Radar
                    name={player.name}
                    dataKey="player"
                    stroke={scoreColor}
                    fill={scoreColor}
                    fillOpacity={0.25}
                    strokeWidth={2}
                  />
                  <Radar
                    name={`${player.position} Avg`}
                    dataKey="posAvg"
                    stroke="#475569"
                    fill="#475569"
                    fillOpacity={0.15}
                    strokeWidth={1.5}
                    strokeDasharray="4 2"
                  />
                  <Legend
                    wrapperStyle={{ fontSize: '12px', color: 'var(--color-muted-foreground)', paddingTop: '8px' }}
                  />
                  <Tooltip
                    contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-border)', borderRadius: '8px', color: 'var(--color-foreground)', fontSize: '12px' }}
                    labelStyle={{ color: 'var(--color-muted-foreground)' }}
                  />
                </RadarChart>
              </ResponsiveContainer>

              {/* Combine metrics table */}
              <div className="grid grid-cols-3 gap-2 mt-3">
                {[
                  { label: '40-Yard Dash', val: player.combine.fortyYard, unit: 's', good: 'lower' },
                  { label: 'Vertical Jump', val: player.combine.vertical, unit: '"', good: 'higher' },
                  { label: 'Broad Jump', val: player.combine.broadJump, unit: '"', good: 'higher' },
                  { label: '3-Cone Drill', val: player.combine.threeCone, unit: 's', good: 'lower' },
                  { label: 'Short Shuttle', val: player.combine.shuttle, unit: 's', good: 'lower' },
                  { label: 'Bench Press', val: player.combine.benchPress, unit: ' reps', good: 'higher' },
                ].filter(m => m.val != null).map(m => (
                  <div key={m.label} className="bg-secondary/60 rounded-lg p-2.5">
                    <div className="text-muted-foreground text-xs">{m.label}</div>
                    <div className="text-foreground text-sm mt-0.5" style={{ fontWeight: 700 }}>
                      {m.val}{m.unit}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-40 text-muted-foreground">
              <AlertCircle className="w-5 h-5 mr-2" />
              <span>No combine data available</span>
            </div>
          )}
        </Section>

        {/* Feature Importance */}
        <Section title="What Drives This Projection">
          <div className="space-y-5">
            {sortedFeatures.map(([key, value]) => (
              <div key={key}>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-foreground text-sm" style={{ fontWeight: 600 }}>{featureLabels[key]}</span>
                  <span className="text-muted-foreground text-sm" style={{ fontWeight: 700 }}>{value}%</span>
                </div>
                <div className="h-3 bg-secondary rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{ width: `${value}%`, backgroundColor: featureColors[key] }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Primary Driver Highlight */}
          <div className="mt-8 bg-secondary/30 rounded-xl p-4 border border-border/50 flex flex-col justify-center">
            <span className="text-xs text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1.5" style={{ fontWeight: 600 }}>
              <TrendingUp className="w-3.5 h-3.5" />
              Primary Analytics Driver
            </span>
            <div className="flex items-center gap-3">
              <div
                className="w-1.5 h-10 rounded-full flex-shrink-0"
                style={{ backgroundColor: featureColors[sortedFeatures[0][0]] }}
              />
              <div>
                <span className="text-foreground text-sm" style={{ fontWeight: 700 }}>{featureLabels[sortedFeatures[0][0]]}</span>
                <p className="text-xs text-muted-foreground mt-[2px] leading-relaxed">
                  Accounts for the largest share ({sortedFeatures[0][1]}%) of {player.name.split(' ')[0]}'s overall prediction relative to baseline.
                </p>
              </div>
            </div>
          </div>

          {/* Model confidence */}
          <div className="mt-4 p-3 bg-secondary/60 rounded-lg border border-border/50 flex items-start gap-2">
            <Award className="w-4 h-4 text-muted-foreground flex-shrink-0 mt-0.5" />
            <p className="text-muted-foreground text-xs leading-relaxed">{confidence}</p>
          </div>
        </Section>
      </div>
    </div>
  );
}
