import { useEffect, useState, useMemo } from 'react';
import { useSearchParams, useNavigate } from 'react-router';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Legend, Tooltip,
} from 'recharts';
import { ChevronDown, Users, ArrowLeft, Trophy, Minus, User, Loader2 } from 'lucide-react';
import {
  fetchPlayers, Player, formatHeight, formatPick, getScoreColor,
  getTierColors,
} from '../data/players';
import { FilterDropdown } from '../components/ui/filter-dropdown';

const RADAR_METRICS = [
  { metric: 'Speed', key: 'speed' },
  { metric: 'Burst', key: 'burst' },
  { metric: 'Agility', key: 'agility' },
  { metric: 'Power', key: 'power' },
  { metric: 'Frame', key: 'frame' },
  { metric: 'Stature', key: 'stature' },
];

const PLAYER_COLORS = ['var(--color-success)', 'var(--color-primary)'] as const;

function PlayerSelect({
  value,
  onChange,
  excludeId,
  label,
  accentColor,
  playersList,
}: {
  value: string;
  onChange: (id: string) => void;
  excludeId: string;
  label: string;
  accentColor: string;
  playersList: Player[];
}) {
  const options = [
    { label: '— Select a player —', value: '' },
    ...playersList
      .filter(p => p.id !== excludeId)
      .sort((a, b) => b.sleeperScore - a.sleeperScore)
      .map(p => ({
        label: p.name,
        value: p.id,
        description: `${p.position}, Rd ${p.draftRound} Pick ${p.draftPick} · Score: ${p.sleeperScore}`
      }))
  ];

  return (
    <div className="flex-1 w-full group" style={{ '--form-accent': accentColor, '--primary': accentColor } as React.CSSProperties}>
      <FilterDropdown
        label={label}
        icon={<User className="h-5 w-5" style={{ color: accentColor }} />}
        value={value}
        options={options}
        onChange={onChange}
        accentColor={accentColor}
        colorAllOptions={label === 'Player 1'}
      />
    </div>
  );
}

function MiniStatCard({ label, value, color, sub }: { label: string; value: string; color: string; sub?: string }) {
  return (
    <div className="bg-secondary rounded-lg p-3 text-center">
      <div className="text-muted-foreground text-xs mb-1" style={{ textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>{label}</div>
      <div style={{ color, fontWeight: 800, fontSize: '1.4rem', lineHeight: 1 }}>{value}</div>
      {sub && <div className="text-muted-foreground text-xs mt-1">{sub}</div>}
    </div>
  );
}

function ProfileCard({ player, opponent, color, isWinner }: { player: Player; opponent?: Player; color: string; isWinner?: boolean }) {
  const { bg, text, border } = getTierColors(player.tier);

  const getCol = (val: number, oppVal?: number, higherIsBetter = true, defColor = color) => {
    if (oppVal === undefined) return defColor;
    return winnerOf(val, oppVal, higherIsBetter) === 'player1' ? color : 'var(--color-muted-foreground)';
  };

  return (
    <div 
      className="bg-card border rounded-xl p-5 relative overflow-hidden transition-all duration-300" 
      style={{ 
        borderTopColor: color, 
        borderTopWidth: 3,
        borderColor: isWinner ? color : 'var(--color-border)',
        boxShadow: isWinner ? `0 0 25px -5px ${color}60` : undefined,
      }}
    >
      {isWinner && (
        <div className="absolute top-0 left-1/2 -translate-x-1/2 py-1 px-5 flex items-center gap-1.5 shadow-sm rounded-b-xl" style={{ backgroundColor: color, color: 'var(--color-background)' }}>
          <Trophy className="w-3.5 h-3.5" />
          <span className="text-[10px] font-bold uppercase tracking-widest">Winner</span>
        </div>
      )}

      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="text-xl" style={{ fontWeight: 800, color }}>{player.name}</div>
          <div className="flex items-center gap-2 mt-1 flex-wrap">
            <span className="bg-secondary text-xs px-2 py-0.5 rounded" style={{ fontWeight: 700, color }}>{player.position}</span>
            <span className="text-muted-foreground text-sm">{player.college}</span>
          </div>
          <div className="text-muted-foreground text-sm mt-1">{formatPick(player.draftRound, player.draftPick)}</div>
          <div className="text-muted-foreground text-sm">{formatHeight(player.height)} · {player.weight} lbs</div>
        </div>
        <div
          className="w-14 h-14 rounded-xl flex flex-col items-center justify-center flex-shrink-0"
          style={{ background: `${color}15`, border: `2px solid ${color}40` }}
        >
          <div style={{ color, fontWeight: 900, fontSize: '1.5rem', lineHeight: 1 }}>{player.sleeperScore}</div>
          <div className="text-xs mt-0.5" style={{ color, opacity: 0.7, fontWeight: 600 }}>SCORE</div>
        </div>
      </div>

      {/* Tier */}
      <span className={`inline-flex items-center px-3 py-1 rounded-lg text-sm ${bg} ${text} border ${border} mb-4`} style={{ fontWeight: 700 }}>
        {player.tier}
      </span>

      {/* Key metrics */}
      <div className="grid grid-cols-2 gap-2">
        <MiniStatCard
          label="Surplus"
          value={`${player.surplusValue >= 0 ? '+' : ''}${player.surplusValue}`}
          color={getCol(player.surplusValue, opponent?.surplusValue, true, player.surplusValue >= 0 ? 'var(--color-success)' : 'var(--color-destructive)')}
          sub="picks vs. proj."
        />
        <MiniStatCard
          label="Breakout"
          value={`${player.breakoutProbability}%`}
          color={getCol(player.breakoutProbability, opponent?.breakoutProbability, true, player.breakoutProbability >= 30 ? 'var(--color-success)' : player.breakoutProbability >= 15 ? 'var(--color-warning)' : 'var(--color-destructive)')}
          sub="probability"
        />
        <MiniStatCard
          label="Durability"
          value={`${player.availabilityFactor}%`}
          color={getCol(player.availabilityFactor, opponent?.availabilityFactor, true, player.availabilityFactor >= 75 ? 'var(--color-success)' : player.availabilityFactor >= 55 ? 'var(--color-warning)' : 'var(--color-destructive)')}
          sub="availability"
        />
        <MiniStatCard
          label="Draft Pick"
          value={`#${player.draftPick}`}
          color={getCol(player.draftPick, opponent?.draftPick, false, color)}
          sub={`Round ${player.draftRound}`}
        />
      </div>

      {/* Combine table */}
      <div className="mt-4 space-y-1.5">
        <div className="text-xs text-muted-foreground mb-2" style={{ textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Combine Metrics</div>
        {[
          { label: '40-Yard Dash', val: player.combine.fortyYard, unit: 's' },
          { label: 'Vertical Jump', val: player.combine.vertical, unit: '"' },
          { label: 'Broad Jump', val: player.combine.broadJump, unit: '"' },
          { label: '3-Cone Drill', val: player.combine.threeCone, unit: 's' },
          { label: 'Short Shuttle', val: player.combine.shuttle, unit: 's' },
          { label: 'Bench Press', val: player.combine.benchPress, unit: ' reps' },
        ].filter(m => m.val != null).map(m => (
          <div key={m.label} className="flex justify-between items-center py-1 border-b border-border/80">
            <span className="text-muted-foreground text-xs">{m.label}</span>
            <span className="text-sm" style={{ fontWeight: 700, color }}>{m.val}{m.unit}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

type WinnerKey = 'player1' | 'player2' | 'tie';

function winnerOf(v1: number, v2: number, higherIsBetter = true): WinnerKey {
  const diff = higherIsBetter ? v1 - v2 : v2 - v1;
  if (diff > 1.5) return 'player1';
  if (diff < -1.5) return 'player2';
  return 'tie';
}

function HeadToHeadRow({
  label,
  v1,
  v2,
  winner,
  p1Color,
  p2Color,
}: {
  label: string;
  v1: string;
  v2: string;
  winner: WinnerKey;
  p1Color: string;
  p2Color: string;
}) {
  return (
    <tr className="border-b border-border/60">
      <td className="py-3 px-4 text-right">
        <span
          className="text-sm"
          style={{
            fontWeight: winner === 'player1' ? 800 : 500,
            color: winner === 'player1' ? p1Color : '#64748b',
          }}
        >
          {v1}
          {winner === 'player1' && <Trophy className="inline w-3.5 h-3.5 ml-1.5" style={{ color: p1Color }} />}
        </span>
      </td>
      <td className="py-3 px-4 text-center">
        <span className="text-muted-foreground text-xs" style={{ textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>{label}</span>
      </td>
      <td className="py-3 px-4 text-left">
        <span
          className="text-sm"
          style={{
            fontWeight: winner === 'player2' ? 800 : 500,
            color: winner === 'player2' ? p2Color : '#64748b',
          }}
        >
          {winner === 'player2' && <Trophy className="inline w-3.5 h-3.5 mr-1.5" style={{ color: p2Color }} />}
          {v2}
        </span>
      </td>
    </tr>
  );
}

export function CompareView() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [playersList, setPlayersList] = useState<Player[]>([]);
  const [loading, setLoading] = useState(true);
  const [p1Id, setP1Id] = useState(searchParams.get('p1') || '');
  const [p2Id, setP2Id] = useState(searchParams.get('p2') || '');

  useEffect(() => {
    fetchPlayers().then(data => {
      setPlayersList(data);
      setLoading(false);
    });
  }, []);

  const p1 = playersList.find(p => p.id === p1Id) || null;
  const p2 = playersList.find(p => p.id === p2Id) || null;

  // Overlaid radar chart data
  const radarData = useMemo(() => {
    if (!p1 || !p2) return [];
    return RADAR_METRICS.map(m => ({
      metric: m.metric,
      [p1.name]: p1.athleticRadar[m.key as keyof typeof p1.athleticRadar] ?? 0,
      [p2.name]: p2.athleticRadar[m.key as keyof typeof p2.athleticRadar] ?? 0,
    }));
  }, [p1, p2]);

  // Overall match results
  const matchResult = useMemo(() => {
    if (!p1 || !p2) return null;
    const dims = [
      winnerOf(p1.sleeperScore, p2.sleeperScore),
      winnerOf(p1.surplusValue, p2.surplusValue),
      winnerOf(p1.breakoutProbability, p2.breakoutProbability),
      winnerOf(p1.availabilityFactor, p2.availabilityFactor),
      winnerOf(p1.draftPick, p2.draftPick, false),
      ...(p1.combine.fortyYard && p2.combine.fortyYard ? [winnerOf(p1.combine.fortyYard, p2.combine.fortyYard, false)] : []),
      ...(p1.combine.vertical && p2.combine.vertical ? [winnerOf(p1.combine.vertical, p2.combine.vertical)] : []),
      ...(p1.combine.broadJump && p2.combine.broadJump ? [winnerOf(p1.combine.broadJump, p2.combine.broadJump)] : []),
    ];
    const p1Wins = dims.filter(d => d === 'player1').length;
    const p2Wins = dims.filter(d => d === 'player2').length;
    const ties = dims.filter(d => d === 'tie').length;
    const overallWinner = p1Wins > p2Wins ? p1 : p2Wins > p1Wins ? p2 : null;
    const overallColor = p1Wins > p2Wins ? PLAYER_COLORS[0] : PLAYER_COLORS[1];

    return { p1Wins, p2Wins, ties, overallWinner, overallColor };
  }, [p1, p2]);

  return (
    <div className="min-h-full bg-background p-6">
      {/* Header */}
      <div className="flex items-center gap-3 mb-5">
        <button
          onClick={() => navigate('/')}
          className="group flex items-center gap-1.5 text-muted-foreground hover:text-foreground text-sm transition-colors"
        >
          <ArrowLeft className="w-4 h-4 text-muted-foreground group-hover:text-[#ccff00] transition-colors" />
          Dashboard
        </button>
        <span className="text-muted-foreground">/</span>
        <div className="flex items-center gap-2">
          <Users className="w-4 h-4 text-[#ccff00]" />
          <span className="text-foreground text-sm" style={{ fontWeight: 600 }}>Compare Players</span>
        </div>
      </div>

      <h1 className="text-foreground text-2xl mb-1" style={{ fontWeight: 700 }}>Player Comparison</h1>
      <p className="text-muted-foreground text-sm mb-6">Select two prospects to compare their sleeper scores, athletic profiles, and projections side by side.</p>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-20">
          <Loader2 className="w-8 h-8 text-[#ccff00] animate-spin mb-4" />
          <p className="text-muted-foreground">Loading draft prospects...</p>
        </div>
      ) : (
        <>
          {/* Player Selectors */}
          <div className="bg-card border border-border rounded-xl p-5 mb-6">
            <div className="flex flex-wrap items-center gap-4">
              <PlayerSelect
                value={p1Id}
                onChange={setP1Id}
                excludeId={p2Id}
                label="Player 1"
                accentColor={PLAYER_COLORS[0]}
                playersList={playersList}
              />
              <div className="flex items-center justify-center w-8 h-8 rounded-full bg-secondary border border-border flex-shrink-0">
                <span className="text-muted-foreground text-xs" style={{ fontWeight: 700 }}>vs</span>
              </div>
              <PlayerSelect
                value={p2Id}
                onChange={setP2Id}
                excludeId={p1Id}
                label="Player 2"
                accentColor={PLAYER_COLORS[1]}
                playersList={playersList}
              />
            </div>
          </div>

          {/* Empty state */}
          {(!p1 || !p2) && (
            <div className="bg-card border border-border rounded-xl p-16 text-center">
              <Users className="w-12 h-12 text-[#ccff00] mx-auto mb-4" />
              <p className="text-primary text-lg" style={{ fontWeight: 600 }}>Select two players to compare</p>
              <p className="text-primary text-sm mt-2">Use the dropdowns above to choose two prospects.</p>
            </div>
          )}

          {/* Comparison content */}
          {p1 && p2 && (
            <div className="space-y-5">

              {/* Side-by-side profile cards */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                <ProfileCard player={p1} opponent={p2} color={PLAYER_COLORS[0]} isWinner={matchResult?.overallWinner === p1} />
                <ProfileCard player={p2} opponent={p1} color={PLAYER_COLORS[1]} isWinner={matchResult?.overallWinner === p2} />
              </div>

              {/* Overlaid Radar Chart */}
              <div className="bg-card border border-border rounded-xl p-5">
                <h3 className="text-muted-foreground text-xs uppercase tracking-widest mb-4" style={{ fontWeight: 600 }}>Athletic Profile Overlay</h3>
                <ResponsiveContainer width="100%" height={340}>
                  <RadarChart data={radarData} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
                    <PolarGrid stroke="var(--color-border)" />
                    <PolarAngleAxis
                      dataKey="metric"
                      tick={{ fill: 'var(--color-muted-foreground)', fontSize: 12, fontWeight: 600 }}
                    />
                    <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                    <Radar
                      name={p1.name}
                      dataKey={p1.name}
                      stroke={PLAYER_COLORS[0]}
                      fill={PLAYER_COLORS[0]}
                      fillOpacity={0.2}
                      strokeWidth={2}
                    />
                    <Radar
                      name={p2.name}
                      dataKey={p2.name}
                      stroke={PLAYER_COLORS[1]}
                      fill={PLAYER_COLORS[1]}
                      fillOpacity={0.2}
                      strokeWidth={2}
                    />
                    <Legend
                      wrapperStyle={{ fontSize: '12px', color: 'var(--color-muted-foreground)', paddingTop: '12px' }}
                    />
                    <Tooltip
                      contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-border)', borderRadius: '8px', color: 'var(--color-foreground)', fontSize: '12px' }}
                    />
                  </RadarChart>
                </ResponsiveContainer>

                {/* Radar dimension breakdown */}
                <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 mt-4 pt-4 border-t border-border">
                  {RADAR_METRICS.map(m => {
                    const v1 = p1.athleticRadar[m.key as keyof typeof p1.athleticRadar] as number ?? 0;
                    const v2 = p2.athleticRadar[m.key as keyof typeof p2.athleticRadar] as number ?? 0;
                    const w = winnerOf(v1, v2);
                    return (
                      <div key={m.key} className="bg-secondary rounded-lg p-2.5 text-center">
                        <div className="text-muted-foreground text-xs mb-2">{m.metric}</div>
                        <div className="flex items-center justify-center gap-1.5">
                          <span
                            className="text-sm"
                            style={{ fontWeight: w === 'player1' ? 800 : 500, color: w === 'player1' ? PLAYER_COLORS[0] : '#64748b' }}
                          >
                            {v1}
                          </span>
                          <span className="text-muted-foreground text-xs">vs</span>
                          <span
                            className="text-sm"
                            style={{ fontWeight: w === 'player2' ? 800 : 500, color: w === 'player2' ? PLAYER_COLORS[1] : '#64748b' }}
                          >
                            {v2}
                          </span>
                        </div>
                        {w !== 'tie' && (
                          <div
                            className="text-xs mt-1"
                            style={{ color: w === 'player1' ? PLAYER_COLORS[0] : PLAYER_COLORS[1], fontWeight: 700 }}
                          >
                            {w === 'player1' ? (p1.name.split(' ').slice(1).join(' ') || p1.name) : (p2.name.split(' ').slice(1).join(' ') || p2.name)} wins
                          </div>
                        )}
                        {w === 'tie' && <div className="text-muted-foreground text-xs mt-1">Even</div>}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Head-to-Head Table */}
              <div className="bg-card border border-border rounded-xl overflow-hidden">
                <div className="px-5 py-4 border-b border-border">
                  <h3 className="text-muted-foreground text-xs uppercase tracking-widest" style={{ fontWeight: 600 }}>Head-to-Head Summary</h3>
                </div>

                {/* Column headers */}
                <div className="grid grid-cols-3 border-b border-border">
                  <div className="py-3 px-4 text-right">
                    <span style={{ color: PLAYER_COLORS[0], fontWeight: 700, fontSize: '13px' }}>{p1.name}</span>
                  </div>
                  <div className="py-3 px-4 text-center">
                    <span className="text-muted-foreground text-xs" style={{ textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Category</span>
                  </div>
                  <div className="py-3 px-4 text-left">
                    <span style={{ color: PLAYER_COLORS[1], fontWeight: 700, fontSize: '13px' }}>{p2.name}</span>
                  </div>
                </div>

                <table className="w-full">
                  <tbody>
                    <HeadToHeadRow
                      label="Sleeper Score"
                      v1={String(p1.sleeperScore)}
                      v2={String(p2.sleeperScore)}
                      winner={winnerOf(p1.sleeperScore, p2.sleeperScore)}
                      p1Color={PLAYER_COLORS[0]}
                      p2Color={PLAYER_COLORS[1]}
                    />
                    <HeadToHeadRow
                      label="Surplus Value"
                      v1={`${p1.surplusValue >= 0 ? '+' : ''}${p1.surplusValue}`}
                      v2={`${p2.surplusValue >= 0 ? '+' : ''}${p2.surplusValue}`}
                      winner={winnerOf(p1.surplusValue, p2.surplusValue)}
                      p1Color={PLAYER_COLORS[0]}
                      p2Color={PLAYER_COLORS[1]}
                    />
                    <HeadToHeadRow
                      label="Breakout Probability"
                      v1={`${p1.breakoutProbability}%`}
                      v2={`${p2.breakoutProbability}%`}
                      winner={winnerOf(p1.breakoutProbability, p2.breakoutProbability)}
                      p1Color={PLAYER_COLORS[0]}
                      p2Color={PLAYER_COLORS[1]}
                    />
                    <HeadToHeadRow
                      label="Durability / Availability"
                      v1={`${p1.availabilityFactor}%`}
                      v2={`${p2.availabilityFactor}%`}
                      winner={winnerOf(p1.availabilityFactor, p2.availabilityFactor)}
                      p1Color={PLAYER_COLORS[0]}
                      p2Color={PLAYER_COLORS[1]}
                    />
                    <HeadToHeadRow
                      label="Draft Pick (Overall)"
                      v1={`#${p1.draftPick}`}
                      v2={`#${p2.draftPick}`}
                      winner={winnerOf(p1.draftPick, p2.draftPick, false)} // lower is better
                      p1Color={PLAYER_COLORS[0]}
                      p2Color={PLAYER_COLORS[1]}
                    />
                    {p1.combine.fortyYard && p2.combine.fortyYard && (
                      <HeadToHeadRow
                        label="40-Yard Dash"
                        v1={`${p1.combine.fortyYard}s`}
                        v2={`${p2.combine.fortyYard}s`}
                        winner={winnerOf(p1.combine.fortyYard, p2.combine.fortyYard, false)}
                        p1Color={PLAYER_COLORS[0]}
                        p2Color={PLAYER_COLORS[1]}
                      />
                    )}
                    {p1.combine.vertical && p2.combine.vertical && (
                      <HeadToHeadRow
                        label="Vertical Jump"
                        v1={`${p1.combine.vertical}"`}
                        v2={`${p2.combine.vertical}"`}
                        winner={winnerOf(p1.combine.vertical, p2.combine.vertical)}
                        p1Color={PLAYER_COLORS[0]}
                        p2Color={PLAYER_COLORS[1]}
                      />
                    )}
                    {p1.combine.broadJump && p2.combine.broadJump && (
                      <HeadToHeadRow
                        label="Broad Jump"
                        v1={`${p1.combine.broadJump}"`}
                        v2={`${p2.combine.broadJump}"`}
                        winner={winnerOf(p1.combine.broadJump, p2.combine.broadJump)}
                        p1Color={PLAYER_COLORS[0]}
                        p2Color={PLAYER_COLORS[1]}
                      />
                    )}
                  </tbody>
                </table>

                {/* Win count summary */}
                {matchResult && (
                  <div className="border-t border-border px-5 py-5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-6">
                        <div className="text-center">
                          <div className="text-2xl" style={{ color: PLAYER_COLORS[0], fontWeight: 800 }}>{matchResult.p1Wins}</div>
                          <div className="text-muted-foreground text-xs">{p1.name.split(' ').slice(1).join(' ') || p1.name} wins</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl text-muted-foreground" style={{ fontWeight: 800 }}>{matchResult.ties}</div>
                          <div className="text-muted-foreground text-xs">Ties</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl" style={{ color: PLAYER_COLORS[1], fontWeight: 800 }}>{matchResult.p2Wins}</div>
                          <div className="text-muted-foreground text-xs">{p2.name.split(' ').slice(1).join(' ') || p2.name} wins</div>
                        </div>
                      </div>
                      
                      {matchResult.overallWinner ? (
                        <div className="flex items-center gap-3 bg-secondary/80 border-2 rounded-xl px-4 py-2" style={{ borderColor: matchResult.overallColor }}>
                          <Trophy className="w-7 h-7" style={{ color: matchResult.overallColor }} />
                          <div>
                            <div className="text-[10px] text-muted-foreground uppercase tracking-widest" style={{ fontWeight: 800, marginBottom: '-2px' }}>Overall Winner</div>
                            <div className="text-lg" style={{ color: matchResult.overallColor, fontWeight: 900, lineHeight: 1.1 }}>
                              {matchResult.overallWinner.name}
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2 bg-secondary rounded-lg px-4 py-3">
                          <Minus className="w-5 h-5 text-muted-foreground" />
                          <span className="text-muted-foreground text-sm" style={{ fontWeight: 700 }}>Too close to call</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
