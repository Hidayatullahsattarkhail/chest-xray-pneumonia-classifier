import { ArrowLeft, ShieldPlus } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function NotFound() {
  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center bg-slate-50 px-4">
      {/* Brand mark */}
      <div className="flex items-center gap-2.5 mb-12">
        <div className="bg-teal-500/10 p-2 rounded-lg">
          <ShieldPlus className="h-5 w-5 text-teal-600" />
        </div>
        <span className="font-bold text-slate-800 text-lg tracking-tight">PneumoScan AI</span>
      </div>

      {/* 404 content */}
      <div className="text-center max-w-md">
        <div className="text-8xl font-extrabold text-slate-200 tracking-tight select-none mb-4">
          404
        </div>
        <h1 className="text-2xl font-bold text-slate-900 mb-3">Page not found</h1>
        <p className="text-slate-500 mb-8 leading-relaxed">
          The page you're looking for doesn't exist or has been moved.
          Head back to the analyzer and upload a chest X-ray.
        </p>
        <Button
          size="lg"
          className="gap-2"
          onClick={() => { window.location.href = "/"; }}
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Home
        </Button>
      </div>
    </div>
  );
}
