import { useState } from "react";
import { ShieldPlus, Menu, X } from "lucide-react";

const links = [
  { label: "Analyze", href: "#analyze" },
  { label: "Performance", href: "#performance" },
  { label: "How it Works", href: "#how-it-works" },
  { label: "Samples", href: "#samples" },
  { label: "About", href: "#about" },
];

function smoothScroll(href: string) {
  const id = href.replace("#", "");
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
}

export function Navbar() {
  const [open, setOpen] = useState(false);

  const handleNav = (href: string) => {
    smoothScroll(href);
    setOpen(false);
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/80 shadow-sm">
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        {/* Logo */}
        <a
          href="#"
          onClick={(e) => { e.preventDefault(); window.scrollTo({ top: 0, behavior: "smooth" }); }}
          className="flex items-center gap-2.5 group"
        >
          <div className="bg-teal-500/10 p-2 rounded-lg group-hover:bg-teal-500/20 transition-colors">
            <ShieldPlus className="h-5 w-5 text-teal-600" />
          </div>
          <div>
            <div className="text-lg font-bold tracking-tight text-slate-900 leading-none">PneumoScan AI</div>
            <div className="text-[9px] uppercase font-semibold tracking-widest text-slate-400 mt-0.5">Clinical Demo Tool</div>
          </div>
        </a>

        {/* Desktop nav */}
        <nav className="hidden md:flex items-center gap-1">
          {links.map((l) => (
            <a
              key={l.label}
              href={l.href}
              onClick={(e) => { e.preventDefault(); smoothScroll(l.href); }}
              className="px-3 py-1.5 rounded-md text-sm font-medium text-slate-600 hover:text-teal-600 hover:bg-teal-50 transition-colors duration-150"
            >
              {l.label}
            </a>
          ))}
        </nav>

        {/* Mobile hamburger */}
        <button
          className="md:hidden p-2 rounded-md text-slate-600 hover:text-teal-600 hover:bg-teal-50 transition-colors"
          onClick={() => setOpen((v) => !v)}
          aria-label="Toggle menu"
        >
          {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {/* Mobile drawer */}
      {open && (
        <div className="md:hidden border-t bg-white shadow-lg animate-in slide-in-from-top-2 duration-200">
          <nav className="container mx-auto px-4 py-3 flex flex-col gap-1">
            {links.map((l) => (
              <a
                key={l.label}
                href={l.href}
                onClick={(e) => { e.preventDefault(); handleNav(l.href); }}
                className="px-3 py-2.5 rounded-md text-sm font-medium text-slate-700 hover:text-teal-600 hover:bg-teal-50 transition-colors"
              >
                {l.label}
              </a>
            ))}
          </nav>
        </div>
      )}
    </header>
  );
}
