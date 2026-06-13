import { BarChart2, Target, Activity, Award } from "lucide-react";

const stats = [
  {
    label: "Test Accuracy",
    value: "92%",
    description: "Overall correct predictions on held-out test set",
    icon: Award,
    color: "text-teal-600",
    bg: "bg-teal-50",
    bar: "bg-teal-500",
    pct: 92,
  },
  {
    label: "Precision",
    value: "90%",
    description: "Of all PNEUMONIA predictions, 90% were correct",
    icon: Target,
    color: "text-blue-600",
    bg: "bg-blue-50",
    bar: "bg-blue-500",
    pct: 90,
  },
  {
    label: "Recall",
    value: "94%",
    description: "Of all actual PNEUMONIA cases, 94% were detected",
    icon: Activity,
    color: "text-emerald-600",
    bg: "bg-emerald-50",
    bar: "bg-emerald-500",
    pct: 94,
  },
  {
    label: "F1-Score",
    value: "92%",
    description: "Harmonic mean of precision and recall",
    icon: BarChart2,
    color: "text-violet-600",
    bg: "bg-violet-50",
    bar: "bg-violet-500",
    pct: 92,
  },
];

export function ModelPerformance() {
  return (
    <section id="performance" className="py-20 px-4 bg-white border-b border-slate-100">
      <div className="container mx-auto max-w-5xl">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold tracking-tight text-slate-900 mb-4">
            Model Performance
          </h2>
          <p className="text-slate-600 max-w-xl mx-auto">
            Evaluated on the held-out test set from the Chest X-Ray Pneumonia dataset
            (Kermany et al., Cell 2018).
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((s) => {
            const Icon = s.icon;
            return (
              <div
                key={s.label}
                className="rounded-2xl border border-slate-100 bg-white shadow-sm p-6 flex flex-col gap-4 hover:shadow-md transition-shadow duration-200"
              >
                <div className={`w-10 h-10 rounded-xl ${s.bg} flex items-center justify-center`}>
                  <Icon className={`w-5 h-5 ${s.color}`} />
                </div>

                <div>
                  <div className="text-3xl font-extrabold text-slate-900 tracking-tight">
                    {s.value}
                  </div>
                  <div className="text-sm font-semibold text-slate-700 mt-0.5">{s.label}</div>
                </div>

                {/* Bar */}
                <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${s.bar} rounded-full`}
                    style={{ width: `${s.pct}%` }}
                  />
                </div>

                <p className="text-xs text-slate-500 leading-relaxed">{s.description}</p>
              </div>
            );
          })}
        </div>

        <p className="text-center text-xs text-slate-400 mt-8">
          * Placeholder values — will be updated with exact evaluation results.
          Precision, recall, and F1 computed for the PNEUMONIA class.
        </p>
      </div>
    </section>
  );
}
