import { PlayCircle } from "lucide-react";

export function SampleImages() {
  
  const handleSampleClick = (url: string) => {
    // Call the global function registered by UploadAnalyzer
    if (typeof (window as any).loadSampleToAnalyzer === 'function') {
      (window as any).loadSampleToAnalyzer(url);
      document.getElementById('analyze')?.scrollIntoView({ behavior: 'smooth' });
    }
  };

  // Using simple placeholder data URIs for robust demoing without external dependencies
  const normalImg = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400' viewBox='0 0 400 400'%3E%3Crect width='400' height='400' fill='%231e293b'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' font-family='sans-serif' font-size='24' fill='%2394a3b8'%3ESample Normal X-Ray%3C/text%3E%3C/svg%3E";
  const pneumoniaImg = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400' viewBox='0 0 400 400'%3E%3Crect width='400' height='400' fill='%23334155'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' font-family='sans-serif' font-size='24' fill='%23cbd5e1'%3ESample Pneumonia X-Ray%3C/text%3E%3C/svg%3E";

  return (
    <section id="samples" className="py-20 px-4 bg-slate-50">
      <div className="container mx-auto max-w-4xl text-center">
        <h2 className="text-2xl font-bold tracking-tight text-slate-900 mb-4">Try It Yourself</h2>
        <p className="text-slate-600 mb-10">
          Don't have an X-ray handy? Use these sample images to test the model's capabilities instantly.
        </p>

        <div className="flex flex-col sm:flex-row gap-6 justify-center max-w-2xl mx-auto">
          
          <div 
            className="group relative rounded-xl overflow-hidden shadow-sm border border-slate-200 cursor-pointer flex-1"
            onClick={() => handleSampleClick(normalImg)}
          >
            <div className="aspect-square bg-slate-900 flex items-center justify-center overflow-hidden">
              <img src={normalImg} alt="Normal Chest X-Ray" className="opacity-80 group-hover:opacity-100 transition-opacity duration-300 group-hover:scale-105" />
            </div>
            <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              <div className="bg-white text-slate-900 rounded-full py-2 px-4 font-semibold text-sm flex items-center gap-2">
                <PlayCircle className="w-4 h-4 text-primary" />
                Test Normal
              </div>
            </div>
            <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent p-4 text-left">
              <span className="text-white font-medium text-sm">Normal Baseline</span>
            </div>
          </div>

          <div 
            className="group relative rounded-xl overflow-hidden shadow-sm border border-slate-200 cursor-pointer flex-1"
            onClick={() => handleSampleClick(pneumoniaImg)}
          >
            <div className="aspect-square bg-slate-800 flex items-center justify-center overflow-hidden">
              <img src={pneumoniaImg} alt="Pneumonia Chest X-Ray" className="opacity-80 group-hover:opacity-100 transition-opacity duration-300 group-hover:scale-105" />
            </div>
            <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              <div className="bg-white text-slate-900 rounded-full py-2 px-4 font-semibold text-sm flex items-center gap-2">
                <PlayCircle className="w-4 h-4 text-destructive" />
                Test Pneumonia
              </div>
            </div>
            <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent p-4 text-left">
              <span className="text-white font-medium text-sm">Pneumonia Positive</span>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
}
