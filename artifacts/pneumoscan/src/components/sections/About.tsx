import { Info } from "lucide-react";

export function About() {
  return (
    <section id="about" className="py-20 px-4 bg-white border-b border-slate-100">
      <div className="container mx-auto max-w-3xl text-center">
        <div className="inline-flex items-center justify-center p-3 bg-primary/10 rounded-full mb-6">
          <Info className="w-6 h-6 text-primary" />
        </div>
        <h2 className="text-3xl font-bold tracking-tight text-slate-900 mb-6">About PneumoScan AI</h2>
        <p className="text-lg text-slate-600 leading-relaxed mb-6">
          PneumoScan AI was developed as a clinical-grade demonstration of how deep learning can assist in radiological workflows. By utilizing transfer learning with a DenseNet121 architecture, the model learns intricate patterns associated with pneumonia from a dataset of 5,856 verified pediatric chest X-rays.
        </p>
        <p className="text-lg text-slate-600 leading-relaxed">
          The goal of this project is to provide a transparent, accessible, and fast AI inference endpoint that researchers, medical students, and developers can use to understand the capabilities and limitations of computer vision in medical diagnostics.
        </p>
      </div>
    </section>
  );
}
