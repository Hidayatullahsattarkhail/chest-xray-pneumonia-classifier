import { ShieldPlus } from "lucide-react";

export function Navbar() {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="bg-primary/10 p-2 rounded-lg">
            <ShieldPlus className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-foreground leading-none">PneumoScan AI</h1>
            <p className="text-[10px] uppercase font-medium tracking-wider text-muted-foreground mt-0.5">Clinical Demo Tool</p>
          </div>
        </div>
        <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-muted-foreground">
          <a href="#analyze" className="hover:text-primary transition-colors">Analyze</a>
          <a href="#how-it-works" className="hover:text-primary transition-colors">How it Works</a>
          <a href="#samples" className="hover:text-primary transition-colors">Samples</a>
        </nav>
      </div>
    </header>
  );
}
