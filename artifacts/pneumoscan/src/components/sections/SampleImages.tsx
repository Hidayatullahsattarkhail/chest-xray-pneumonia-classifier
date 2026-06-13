import { PlayCircle, ImageOff } from "lucide-react";
import { useState } from "react";

// Public-domain chest X-ray images from Wikimedia Commons (CORS-accessible)
const SAMPLES = [
  {
    label: "Normal",
    sublabel: "Healthy lung fields",
    badge: "NORMAL",
    badgeClass: "bg-green-100 text-green-700 border-green-200",
    // Normal PA chest radiograph — Wikimedia Commons, public domain
    url: "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg/480px-Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg",
  },
  {
    label: "Pneumonia",
    sublabel: "Bilateral infiltrates visible",
    badge: "PNEUMONIA",
    badgeClass: "bg-red-100 text-red-700 border-red-200",
    // Pneumococcal pneumonia chest X-ray — Wikimedia Commons, public domain
    url: "https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Pneumococcal_pneumonia.jpg/480px-Pneumococcal_pneumonia.jpg",
  },
];

export function SampleImages() {
  const [loading, setLoading] = useState<string | null>(null);
  const [errors, setErrors] = useState<Record<string, boolean>>({});

  const handleSampleClick = async (sample: typeof SAMPLES[0]) => {
    if (loading) return;
    setLoading(sample.label);
    try {
      const response = await fetch(sample.url);
      if (!response.ok) throw new Error("Fetch failed");
      const blob = await response.blob();
      const ext = blob.type === "image/png" ? "png" : "jpg";
      const file = new File([blob], `sample-xray-${sample.label.toLowerCase()}.${ext}`, {
        type: blob.type || "image/jpeg",
      });

      const loader = (window as any).loadSampleToAnalyzer;
      if (typeof loader === "function") {
        // Pass file directly instead of URL to avoid double-fetch
        loader(file);
        document.getElementById("analyze")?.scrollIntoView({ behavior: "smooth" });
      }
    } catch {
      setErrors((e) => ({ ...e, [sample.label]: true }));
    } finally {
      setLoading(null);
    }
  };

  return (
    <section id="samples" className="py-20 px-4 bg-slate-50 border-b border-slate-100">
      <div className="container mx-auto max-w-4xl">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold tracking-tight text-slate-900 mb-4">
            Try a Sample Image
          </h2>
          <p className="text-slate-600 max-w-xl mx-auto">
            Don't have an X-ray handy? Click one of these public-domain chest X-rays to
            load it into the analyzer and run a prediction instantly.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row gap-6 justify-center max-w-2xl mx-auto">
          {SAMPLES.map((s) => {
            const isLoading = loading === s.label;
            const hasError = errors[s.label];

            return (
              <div
                key={s.label}
                onClick={() => !hasError && handleSampleClick(s)}
                className={`group relative rounded-2xl overflow-hidden shadow-sm border border-slate-200 flex-1 cursor-pointer
                  hover:shadow-lg hover:border-teal-300 transition-all duration-200
                  ${isLoading ? "opacity-70 cursor-wait" : ""}
                  ${hasError ? "opacity-50 cursor-not-allowed" : ""}`}
              >
                {/* Image */}
                <div className="aspect-square bg-slate-900 flex items-center justify-center overflow-hidden">
                  {hasError ? (
                    <div className="flex flex-col items-center gap-2 text-slate-500">
                      <ImageOff className="w-8 h-8" />
                      <span className="text-xs">Image unavailable</span>
                    </div>
                  ) : (
                    <img
                      src={s.url}
                      alt={`${s.label} chest X-ray`}
                      className="w-full h-full object-cover opacity-80 group-hover:opacity-100 group-hover:scale-105 transition-all duration-300"
                      onError={() => setErrors((e) => ({ ...e, [s.label]: true }))}
                    />
                  )}
                </div>

                {/* Hover overlay */}
                {!hasError && (
                  <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    <div className="bg-white text-slate-900 rounded-full py-2 px-5 font-semibold text-sm flex items-center gap-2 shadow-lg">
                      {isLoading ? (
                        <span className="animate-pulse">Loading...</span>
                      ) : (
                        <>
                          <PlayCircle className="w-4 h-4 text-teal-600" />
                          Analyze this X-Ray
                        </>
                      )}
                    </div>
                  </div>
                )}

                {/* Bottom label */}
                <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-white font-semibold text-sm">{s.label}</div>
                      <div className="text-white/60 text-xs">{s.sublabel}</div>
                    </div>
                    <span className={`text-xs font-bold px-2 py-0.5 rounded-full border ${s.badgeClass}`}>
                      {s.badge}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <p className="text-center text-xs text-slate-400 mt-6">
          Images sourced from Wikimedia Commons (public domain). Results are AI predictions — not clinical diagnoses.
        </p>
      </div>
    </section>
  );
}
