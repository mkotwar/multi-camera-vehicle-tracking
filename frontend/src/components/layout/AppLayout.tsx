import { Link, NavLink } from "react-router-dom";
import type { ReactNode } from "react";
import vinfoLogo from "../../assets/vinfo-logo.jpeg";

type Props = { children: ReactNode };

export function AppLayout({ children }: Props) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <Link to="/" className="brand" aria-label="VinfoAI home">
          <img className="brand-logo" src={vinfoLogo} alt="VinfoAI logo" />
        </Link>
        <nav className="nav">
          <NavLink to="/">Dashboard</NavLink>
          <NavLink to="/vehicles">Vehicles</NavLink>
          <NavLink to="/video-chat">Video Chat</NavLink>
          <NavLink to="/runs">Runs</NavLink>
          <NavLink to="/run-control">Run Control</NavLink>
          <NavLink to="/settings">Settings</NavLink>
          <NavLink to="/system">System</NavLink>
        </nav>
      </header>
      <main className="content">{children}</main>
    </div>
  );
}
