import { Link, NavLink } from "react-router-dom";
import type { ReactNode } from "react";

type Props = { children: ReactNode };

export function AppLayout({ children }: Props) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <Link to="/" className="brand">Vehicle Analytics Console</Link>
        <nav className="nav">
          <NavLink to="/">Dashboard</NavLink>
          <NavLink to="/vehicles">Vehicles</NavLink>
          <NavLink to="/runs">Runs</NavLink>
          <NavLink to="/system">System</NavLink>
        </nav>
      </header>
      <main className="content">{children}</main>
    </div>
  );
}
