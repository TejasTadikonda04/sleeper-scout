import { NavLink, Outlet, useNavigate } from 'react-router';
import { BarChart3, Users, Home, ChevronRight, Activity } from 'lucide-react';

export function Layout() {
  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden">
      {/* Sidebar */}
      <aside className="w-56 flex-shrink-0 bg-card border-r border-border flex flex-col">
        {/* Logo */}
        <div className="px-5 py-5 border-b border-border">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center flex-shrink-0">
              <Activity className="w-4 h-4 text-primary-foreground" strokeWidth={2.5} />
            </div>
            <div>
              <div className="text-foreground text-sm" style={{ fontWeight: 700, letterSpacing: '0.02em' }}>NFL SCOUT</div>
              <div className="text-muted-foreground" style={{ fontSize: '10px', letterSpacing: '0.08em', textTransform: 'uppercase' }}>Combine Analytics</div>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-150 text-sm group ${
                isActive
                  ? 'bg-primary/15 text-primary border border-primary/20'
                  : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
              }`
            }
          >
            <Home className="w-4 h-4 flex-shrink-0" />
            <span>Dashboard</span>
          </NavLink>

          <NavLink
            to="/compare"
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-150 text-sm group ${
                isActive
                  ? 'bg-primary/15 text-primary border border-primary/20'
                  : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
              }`
            }
          >
            <Users className="w-4 h-4 flex-shrink-0" />
            <span>Compare Players</span>
          </NavLink>
        </nav>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-border">
          <div className="text-muted-foreground" style={{ fontSize: '10px', letterSpacing: '0.05em', textTransform: 'uppercase' }}>
            Data: 2024 NFL Draft Class
          </div>
          <div className="text-muted-foreground mt-0.5" style={{ fontSize: '10px' }}>
            42 Prospects Evaluated
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}
