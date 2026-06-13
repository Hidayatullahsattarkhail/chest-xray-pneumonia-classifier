import { Button } from "@/components/ui/button";
import { ArrowRight, Activity, Microscope, ShieldCheck } from "lucide-react";

export function Hero() {
  return (
    <section className="pt-20 pb-16 md:pt-32 md:pb-24 px-4 overflow-hidden relative">
      {/* Abstract background decorative elements */}
      <div className="absolute top-0 inset-x-0 h-full overflow-hidden -z-10 opacity-30 pointer-events-none">
        <div className="absolute -top-40 -right-40 w-96 h-96 rounded-full bg-primary/20 blur-3xl" />
        <div className="absolute top-40 -left-20 w-72 h-72 rounded-full bg-blue-500/10 blur-3xl" />
      </div>

      <div className="container mx-auto max-w-4xl text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-semibold tracking-wide uppercase mb-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <Activity className="w-3.5 h-3.5" />
          <span>Clinical Trial Demo</span>
        </div>
        
        <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight text-slate-900 mb-6 animate-in fade-in slide-in-from-bottom-6 duration-700">
          AI-Powered Pneumonia Detection from <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-blue-600">Chest X-Rays</span>
        </h1>
        
        <p className="text-lg md:text-xl text-slate-600 mb-10 max-w-2xl mx-auto leading-relaxed animate-in fade-in slide-in-from-bottom-8 duration-700 delay-150">
          Upload a chest X-ray image and receive an instant, high-confidence prediction using a customized DenseNet121 model trained on thousands of clinical images.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-in fade-in slide-in-from-bottom-10 duration-700 delay-300">
          <Button size="lg" className="w-full sm:w-auto text-base h-12 px-8 shadow-lg shadow-primary/25" onClick={() => document.getElementById('analyze')?.scrollIntoView({ behavior: 'smooth' })}>
            Try the Analyzer
            <ArrowRight className="ml-2 w-4 h-4" />
          </Button>
          <Button size="lg" variant="outline" className="w-full sm:w-auto text-base h-12 px-8" onClick={() => document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth' })}>
            How it works
          </Button>
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6 pt-10 border-t border-slate-200/60 text-left animate-in fade-in duration-1000 delay-500">
          <div className="flex flex-col gap-2">
            <div className="bg-white w-10 h-10 rounded-lg shadow-sm border border-slate-100 flex items-center justify-center text-primary">
              <Microscope className="w-5 h-5" />
            </div>
            <h3 className="font-semibold text-slate-900">High Accuracy</h3>
            <p className="text-sm text-slate-500">Built on a specialized DenseNet121 architecture, fine-tuned for radiological imagery.</p>
          </div>
          <div className="flex flex-col gap-2">
            <div className="bg-white w-10 h-10 rounded-lg shadow-sm border border-slate-100 flex items-center justify-center text-primary">
              <Activity className="w-5 h-5" />
            </div>
            <h3 className="font-semibold text-slate-900">Instant Results</h3>
            <p className="text-sm text-slate-500">Powered by Hugging Face Spaces for real-time inference and detailed confidence scoring.</p>
          </div>
          <div className="flex flex-col gap-2">
            <div className="bg-white w-10 h-10 rounded-lg shadow-sm border border-slate-100 flex items-center justify-center text-primary">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <h3 className="font-semibold text-slate-900">Secure & Private</h3>
            <p className="text-sm text-slate-500">Images are processed statelessly and are never stored after the analysis is complete.</p>
          </div>
        </div>
      </div>
    </section>
  );
}
