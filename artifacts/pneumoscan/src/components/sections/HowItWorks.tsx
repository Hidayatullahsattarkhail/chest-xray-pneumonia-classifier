import { Card, CardContent } from "@/components/ui/card";
import { BrainCircuit, Database, Server } from "lucide-react";

export function HowItWorks() {
  return (
    <section id="how-it-works" className="py-20 px-4 bg-white border-y border-slate-100">
      <div className="container mx-auto max-w-5xl">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold tracking-tight text-slate-900 mb-4">Under the Hood</h2>
          <p className="text-slate-600 max-w-2xl mx-auto">
            PneumoScan AI leverages state-of-the-art deep learning architectures specifically adapted for radiological imaging.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <Card className="border-none shadow-none bg-slate-50">
            <CardContent className="pt-6">
              <div className="bg-primary/10 w-12 h-12 rounded-xl flex items-center justify-center mb-4">
                <BrainCircuit className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold mb-2">DenseNet121 Architecture</h3>
              <p className="text-slate-600 text-sm leading-relaxed">
                We use transfer learning on DenseNet121, known for its efficiency and strong gradient flow. The model extracts intricate features from X-rays that might be missed by the naked eye.
              </p>
            </CardContent>
          </Card>

          <Card className="border-none shadow-none bg-slate-50">
            <CardContent className="pt-6">
              <div className="bg-blue-500/10 w-12 h-12 rounded-xl flex items-center justify-center mb-4">
                <Database className="w-6 h-6 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Clinical Dataset</h3>
              <p className="text-slate-600 text-sm leading-relaxed">
                Trained on a curated dataset of 5,856 verified pediatric chest X-ray images. The diverse training data allows the model to generalize well across different image qualities and patient demographics.
              </p>
            </CardContent>
          </Card>

          <Card className="border-none shadow-none bg-slate-50">
            <CardContent className="pt-6">
              <div className="bg-emerald-500/10 w-12 h-12 rounded-xl flex items-center justify-center mb-4">
                <Server className="w-6 h-6 text-emerald-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Hugging Face Deployment</h3>
              <p className="text-slate-600 text-sm leading-relaxed">
                The inference engine is deployed securely via Hugging Face Spaces using Gradio. This provides a scalable, stateless API that processes your images securely in real-time.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
}
