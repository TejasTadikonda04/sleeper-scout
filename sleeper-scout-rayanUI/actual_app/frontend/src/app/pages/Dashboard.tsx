import { useEffect, useState, useMemo } from 'react';
import { useNavigate } from 'react-router';
import { ChevronUp, ChevronDown, ChevronsUpDown, Search, SlidersHorizontal, ArrowUpRight, Layers, Target, Loader2 } from 'lucide-react';
import { fetchPlayers, Player, Position, TierLabel, getTierColors, getScoreColor } from '../data/players';
import { FilterDropdown } from '../components/ui/filter-dropdown';
import { SuggestiveSearch } from '../components/ui/suggestive-search';

type SortField = 'name' | 'position' | 'college' | 'draftPick' | 'sleeperScore' | 'surplusValue' | 'breakoutProbability' | 'tier';
type SortDir = 'asc' | 'desc';

const POSITIONS: (Position | 'ALL')[] = ['ALL', 'QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB'];
const ROUNDS = [
  { label: 'All Rounds', value: 0 },
  { label: 'Round 1', value: 1 },
  { label: 'Round 2', value: 2 },
  { label: 'Rounds 3–7', value: 3 },
];
const SCORE_RANGES = [
  { label: 'All Scores', min: 0, max: 100 },
  { label: 'Elite (80+)', min: 80, max: 100 },
  { label: 'Good (65–79)', min: 65, max: 79 },
  { label: 'Average (50–64)', min: 50, max: 64 },
  { label: 'Developmental (<50)', min: 0, max: 49 },
];

function SortIcon({ field, sortField, sortDir }: { field: SortField; sortField: SortField; sortDir: SortDir }) {
  if (sortField !== field) return <ChevronsUpDown className="w-3.5 h-3.5 text-muted-foreground" />;
  return sortDir === 'asc'
    ? <ChevronUp className="w-3.5 h-3.5 text-success" />
    : <ChevronDown className="w-3.5 h-3.5 text-success" />;
}

function SleeperBadge({ score }: { score: number }) {
  const color = getScoreColor(score);
  const bg = score >= 81 ? 'rgba(16,185,129,0.12)' : score >= 66 ? 'rgba(34,197,94,0.12)' : score >= 51 ? 'rgba(234,179,8,0.12)' : score >= 31 ? 'rgba(249,115,22,0.12)' : 'rgba(239,68,68,0.12)';
  return (
    <span
      className="inline-flex items-center justify-center w-11 h-7 rounded-md text-sm"
      style={{ color, background: bg, fontWeight: 700, border: `1px solid ${color}30` }}
    >
      {score}
    </span>
  );
}

function SurplusCell({ value }: { value: number }) {
  const isPos = value >= 0;
  return (
    <span className={`inline-flex items-center gap-0.5 text-sm ${isPos ? 'text-success' : 'text-destructive'}`} style={{ fontWeight: 600 }}>
      {isPos ? '+' : ''}{value}
    </span>
  );
}

function TierBadge({ tier }: { tier: TierLabel }) {
  const { bg, text, border } = getTierColors(tier);
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs ${bg} ${text} border ${border}`} style={{ fontWeight: 600, whiteSpace: 'nowrap' }}>
      {tier}
    </span>
  );
}

export function Dashboard() {
  const navigate = useNavigate();
  const [players, setPlayers] = useState<Player[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [posFilter, setPosFilter] = useState<Position | 'ALL'>('ALL');
  const [roundFilter, setRoundFilter] = useState(0);
  const [scoreRange, setScoreRange] = useState({ min: 0, max: 100 });
  const [sortField, setSortField] = useState<SortField>('sleeperScore');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  useEffect(() => {
    fetchPlayers().then(data => {
      setPlayers(data);
      setLoading(false);
    });
  }, []);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir(d => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDir('desc');
    }
  };

  const filtered = useMemo(() => {
    return players.filter(p => {
      if (posFilter !== 'ALL' && p.position !== posFilter) return false;
      if (roundFilter === 1 && p.draftRound !== 1) return false;
      if (roundFilter === 2 && p.draftRound !== 2) return false;
      if (roundFilter === 3 && p.draftRound < 3) return false;
      if (p.sleeperScore < scoreRange.min || p.sleeperScore > scoreRange.max) return false;
      if (search && !p.name.toLowerCase().includes(search.toLowerCase()) && !p.college.toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    });
  }, [players, posFilter, roundFilter, scoreRange, search]);

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      let av: any = a[sortField as keyof Player];
      let bv: any = b[sortField as keyof Player];
      if (typeof av === 'number' && typeof bv === 'number') {
        return sortDir === 'asc' ? av - bv : bv - av;
      }
      return sortDir === 'asc' ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    });
  }, [filtered, sortField, sortDir]);

  const handleScoreRange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const idx = parseInt(e.target.value);
    setScoreRange({ min: SCORE_RANGES[idx].min, max: SCORE_RANGES[idx].max });
  };

  const tierCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    players.forEach(p => { counts[p.tier] = (counts[p.tier] || 0) + 1; });
    return counts;
  }, [players]);

  if (loading) {
    return (
      <div className="min-h-full bg-background flex flex-col items-center justify-center">
        <Loader2 className="w-8 h-8 text-primary animate-spin mb-4" />
        <p className="text-muted-foreground">Loading draft prospects...</p>
      </div>
    );
  }

  return (
    <div className="min-h-full bg-background p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-foreground text-2xl" style={{ fontWeight: 700 }}>Draft Prospect Leaderboard</h1>
        <p className="text-muted-foreground mt-1 text-sm">2024 NFL Combine Analytics · {players.length} prospects evaluated</p>
      </div>

      {/* Tier Summary Cards */}
      <div className="grid grid-cols-5 gap-3 mb-6">
        {(['High Upside', 'Safe Floor', 'Boom or Bust', 'Developmental', 'Overdrafted'] as TierLabel[]).map(tier => {
          const { bg, text, border } = getTierColors(tier);
          return (
            <button
              key={tier}
              onClick={() => {/* filter by tier - future enhancement */ }}
              className={`p-3 rounded-lg border flex items-center gap-3 ${bg} ${border} text-left`}
            >
              <div className={`text-2xl ${text}`} style={{ fontWeight: 800 }}>{tierCounts[tier] || 0}</div>
              <div className={`text-sm leading-tight ${text}`} style={{ fontWeight: 600, opacity: 0.9 }}>
                {tier}
              </div>
            </button>
          );
        })}
      </div>

      {/* Filters */}
      <div className="bg-card border border-border rounded-xl p-4 mb-4">
        <div className="flex flex-wrap items-center gap-3">
          {/* Search */}
          <SuggestiveSearch
            suggestions={["Search player name...", "Search college...", "Find sleeper picks..."]}
            pauseAfterTypeMs={4000} // Time to wait after typing (2 seconds)
            typeDurationMs={1000}
            deleteDurationMs={700}
            onChange={setSearch}
            showCursor={false}
            className="bg-secondary w-[240px] rounded-lg h-[38px] border-border"
          />

          {/* Position pills */}
          <div className="flex items-center gap-1.5 flex-wrap">
            {POSITIONS.map(pos => (
              <button
                key={pos}
                onClick={() => setPosFilter(pos)}
                className={`px-2.5 py-1 rounded-md text-xs transition-all ${posFilter === pos
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-secondary text-muted-foreground hover:text-foreground hover:bg-muted'
                  }`}
                style={{ fontWeight: 600 }}
              >
                {pos}
              </button>
            ))}
          </div>

          <div className="ml-auto flex items-center gap-4">
            <FilterDropdown
              label="Draft Round"
              icon={<Layers className="h-4 w-4" />}
              value={roundFilter}
              options={ROUNDS}
              onChange={(val) => setRoundFilter(val as number)}
              searchable={false}
            />

            <FilterDropdown
              label="Sleeper Score"
              icon={<Target className="h-4 w-4" />}
              className="min-w-[260px]"
              value={SCORE_RANGES.findIndex(r => r.min === scoreRange.min && r.max === scoreRange.max)}
              options={SCORE_RANGES.map((r, i) => ({ label: r.label, value: i }))}
              onChange={(val) => setScoreRange({ min: SCORE_RANGES[val].min, max: SCORE_RANGES[val].max })}
              searchable={false}
            />

            <div className="flex flex-col items-center justify-center text-muted-foreground text-xs border-l border-border px-6 min-w-[140px]">
              <span className="flex items-center gap-1 opacity-70 mb-0.5"><SlidersHorizontal className="w-3 h-3" /> Filtered</span>
              <span className="font-semibold text-foreground text-sm">{sorted.length} Players</span>
            </div>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="bg-card border border-border rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[900px]">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left px-4 py-3 text-xs text-muted-foreground uppercase tracking-wider" style={{ fontWeight: 600, width: 36 }}>#</th>
                {[
                  { field: 'name' as SortField, label: 'Player' },
                  { field: 'position' as SortField, label: 'Pos' },
                  { field: 'college' as SortField, label: 'College' },
                  { field: 'draftPick' as SortField, label: 'Draft' },
                  { field: 'sleeperScore' as SortField, label: 'Sleeper Score' },
                  { field: 'surplusValue' as SortField, label: 'Surplus Val.' },
                  { field: 'breakoutProbability' as SortField, label: 'Breakout %' },
                  { field: 'tier' as SortField, label: 'Tier' },
                ].map(col => (
                  <th
                    key={col.field}
                    className="text-left px-4 py-3 text-xs text-muted-foreground uppercase tracking-wider cursor-pointer hover:text-foreground transition-colors select-none"
                    style={{ fontWeight: 600 }}
                    onClick={() => handleSort(col.field)}
                  >
                    <span className="flex items-center gap-1.5">
                      {col.label}
                      <SortIcon field={col.field} sortField={sortField} sortDir={sortDir} />
                    </span>
                  </th>
                ))}
                <th className="px-4 py-3 text-xs text-muted-foreground uppercase tracking-wider w-10" style={{ fontWeight: 600 }}></th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((player, idx) => (
                <tr
                  key={player.id}
                  onClick={() => navigate(`/player/${player.id}`)}
                  className="border-b border-border/60 hover:bg-secondary/50 cursor-pointer transition-colors group"
                >
                  <td className="px-4 py-3 text-muted-foreground text-sm">{idx + 1}</td>
                  <td className="px-4 py-3">
                    <div className="text-foreground text-sm group-hover:text-success transition-colors" style={{ fontWeight: 600 }}>
                      {player.name}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span className="inline-flex items-center justify-center w-8 h-6 rounded text-xs bg-primary text-primary-foreground" style={{ fontWeight: 700 }}>
                      {player.position}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-muted-foreground text-sm">{player.college}</td>
                  <td className="px-4 py-3">
                    <div className="text-foreground text-sm">
                      <span className="text-muted-foreground">R{player.draftRound}</span>
                      <span className="text-muted-foreground mx-1">/</span>
                      <span style={{ fontWeight: 600 }}>#{player.draftPick}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <SleeperBadge score={player.sleeperScore} />
                      {/* Mini bar */}
                      <div className="flex-1 max-w-16 h-1.5 bg-secondary rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${player.sleeperScore}%`, backgroundColor: getScoreColor(player.sleeperScore) }}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <SurplusCell value={player.surplusValue} />
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <span
                        className="text-sm"
                        style={{
                          fontWeight: 600,
                          color: player.breakoutProbability >= 30 ? 'var(--color-success)' : player.breakoutProbability >= 15 ? '#f2a134' : 'var(--color-destructive)'
                        }}
                      >
                        {player.breakoutProbability}%
                      </span>
                      <div className="flex-1 max-w-14 h-1.5 bg-secondary rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${player.breakoutProbability}%`,
                            backgroundColor: player.breakoutProbability >= 30 ? 'var(--color-success)' : player.breakoutProbability >= 15 ? '#f2a134' : 'var(--color-destructive)'
                          }}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <TierBadge tier={player.tier} />
                  </td>
                  <td className="px-4 py-3">
                    <ArrowUpRight className="w-4 h-4 text-muted-foreground group-hover:text-success transition-colors" />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {sorted.length === 0 && (
          <div className="py-16 text-center text-muted-foreground">
            <p className="text-lg" style={{ fontWeight: 600 }}>No players match your filters</p>
            <p className="text-sm mt-1">Try adjusting your search or filter criteria</p>
          </div>
        )}
      </div>
    </div>
  );
}
