import { Github, Linkedin, ExternalLink } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t bg-slate-900 text-slate-300 pt-14 pb-8 mt-20">
      <div className="container mx-auto px-4 max-w-5xl">
        {/* Top row */}
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-10 mb-10">
          {/* Brand */}
          <div className="max-w-xs">
            <h2 className="text-white font-bold text-xl tracking-tight mb-2">PneumoScan AI</h2>
            <p className="text-sm text-slate-400 leading-relaxed">
              An AI-powered clinical demo tool for chest X-ray pneumonia detection.
              Built for educational and research purposes — not intended for clinical use.
            </p>
          </div>

          {/* Links */}
          <div className="flex flex-col gap-3">
            <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-500">Project</h3>
            <a
              href="#"
              className="flex items-center gap-2 text-sm text-slate-400 hover:text-teal-400 transition-colors group"
            >
              <Github className="w-4 h-4 group-hover:scale-110 transition-transform" />
              GitHub Repository
              <ExternalLink className="w-3 h-3 opacity-50" />
            </a>
            <a
              href="#"
              className="flex items-center gap-2 text-sm text-slate-400 hover:text-teal-400 transition-colors group"
            >
              <Linkedin className="w-4 h-4 group-hover:scale-110 transition-transform" />
              LinkedIn Profile
              <ExternalLink className="w-3 h-3 opacity-50" />
            </a>
          </div>

          {/* Disclaimer inline */}
          <div className="max-w-xs">
            <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-3">Disclaimer</h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              This tool is for educational/demonstration purposes only and is NOT a substitute for
              professional medical diagnosis. Always consult a qualified healthcare provider.
            </p>
          </div>
        </div>

        {/* Tech badges */}
        <div className="flex flex-wrap items-center gap-2 border-t border-slate-800 pt-6 pb-4">
          <span className="text-xs text-slate-600 mr-2">Powered by</span>
          {["DenseNet121", "PyTorch", "Hugging Face Spaces", "Gradio", "React"].map((tech) => (
            <span
              key={tech}
              className="px-2.5 py-1 rounded-full bg-slate-800 text-slate-400 text-xs font-medium border border-slate-700"
            >
              {tech}
            </span>
          ))}
        </div>

        {/* Bottom credit */}
        <div className="border-t border-slate-800 pt-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 text-xs text-slate-600">
          <span>
            Built by <span className="text-slate-300 font-medium">Hidayat Ullah</span> &nbsp;|&nbsp; BS Artificial Intelligence, PAF-IAST
          </span>
          <span>© {new Date().getFullYear()} PneumoScan AI</span>
        </div>
      </div>
    </footer>
  );
}
