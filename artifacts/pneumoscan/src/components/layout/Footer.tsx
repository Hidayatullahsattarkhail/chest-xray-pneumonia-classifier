export function Footer() {
  return (
    <footer className="border-t bg-muted/40 py-12 mt-20">
      <div className="container mx-auto px-4 text-center">
        <h2 className="font-semibold text-lg mb-2">PneumoScan AI</h2>
        <p className="text-muted-foreground text-sm max-w-lg mx-auto mb-6">
          This tool is for educational/demonstration purposes only and is NOT a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.
        </p>
        <div className="text-xs text-muted-foreground pt-6 border-t border-border inline-block min-w-[200px]">
          Built by <span className="font-medium text-foreground">Hidayat Ullah</span> | BS AI, PAF-IAST
        </div>
      </div>
    </footer>
  );
}
