const fs = require('fs');
const path = require('path');

const files = [
  'src/app/pages/Dashboard.tsx',
  'src/app/pages/PlayerProfile.tsx',
  'src/app/pages/CompareView.tsx',
];

const replacements = {
  'bg-slate-950': 'bg-background',
  'text-slate-100': 'text-foreground',
  'text-slate-200': 'text-foreground',
  'text-slate-300': 'text-foreground',
  'text-slate-400': 'text-muted-foreground',
  'text-slate-500': 'text-muted-foreground',
  'text-slate-600': 'text-muted-foreground',
  'text-slate-700': 'text-muted-foreground',
  'bg-slate-900': 'bg-card',
  'border-slate-800': 'border-border',
  'border-slate-800/60': 'border-border/60',
  'border-slate-800/80': 'border-border/80',
  'border-slate-700/50': 'border-border/50',
  'border-slate-700': 'border-border',
  'bg-slate-800': 'bg-secondary',
  'bg-slate-800/50': 'bg-secondary/50',
  'bg-slate-800/60': 'bg-secondary/60',
  'hover:bg-slate-800/50': 'hover:bg-secondary/50',
  'placeholder:text-slate-600': 'placeholder:text-muted-foreground',
  'focus:border-emerald-500/60': 'focus:border-primary/60',
  'bg-emerald-500': 'bg-primary',
  'text-slate-950': 'text-primary-foreground',
  'hover:text-slate-200': 'hover:text-foreground',
  'hover:bg-slate-700': 'hover:bg-muted',
  'text-emerald-400': 'text-success',
  'text-red-400': 'text-destructive',
  'hover:text-slate-300': 'hover:text-foreground',
  'border-slate-950': 'border-background',
  'bg-slate-900/60': 'bg-card/60',
  'hover:border-slate-600': 'hover:border-muted-foreground',
  '#0f172a': 'var(--color-card)',
  '#1e293b': 'var(--color-border)',
  '#94a3b8': 'var(--color-muted-foreground)',
  '#f1f5f9': 'var(--color-foreground)',
  '#10b981': 'var(--color-success)',
  '#ef4444': 'var(--color-destructive)',
  '#f59e0b': 'var(--color-warning)',
  '#3b82f6': 'var(--color-primary)', // player 2 color
};

files.forEach(file => {
  const filePath = path.join(__dirname, file);
  if (!fs.existsSync(filePath)) {
    console.error('File not found:', filePath);
    return;
  }
  let content = fs.readFileSync(filePath, 'utf8');
  for (const [key, value] of Object.entries(replacements)) {
    content = content.split(key).join(value);
  }
  fs.writeFileSync(filePath, content);
  console.log('Updated', file);
});
