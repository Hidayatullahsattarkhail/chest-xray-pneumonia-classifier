import { GraduationCap, FlaskConical, Globe } from "lucide-react";

export function About() {
  return (
    <section id="about" className="py-20 px-4 bg-white border-b border-slate-100">
      <div className="container mx-auto max-w-4xl">
        <div className="text-center mb-14">
          <div className="inline-flex items-center justify-center p-3 bg-teal-50 rounded-full mb-5">
            <GraduationCap className="w-6 h-6 text-teal-600" />
          </div>
          <h2 className="text-3xl font-bold tracking-tight text-slate-900 mb-4">About This Project</h2>
          <p className="text-slate-500 max-w-xl mx-auto">
            The story behind PneumoScan AI — from research question to deployed model.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-14">
          <div className="flex flex-col gap-3">
            <div className="w-10 h-10 rounded-xl bg-teal-50 flex items-center justify-center">
              <GraduationCap className="w-5 h-5 text-teal-600" />
            </div>
            <h3 className="font-semibold text-slate-900">Origin</h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Capstone-adjacent project for a BS in Artificial Intelligence at PAF-IAST,
              exploring the practical application of computer vision in radiological diagnosis.
            </p>
          </div>
          <div className="flex flex-col gap-3">
            <div className="w-10 h-10 rounded-xl bg-blue-50 flex items-center justify-center">
              <FlaskConical className="w-5 h-5 text-blue-600" />
            </div>
            <h3 className="font-semibold text-slate-900">Methodology</h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Transfer learning on DenseNet121 pre-trained on ImageNet, fine-tuned on 5,856
              pediatric chest X-rays (Kermany et al.). Grad-CAM was used during development
              to interpret and validate the model's attention regions.
            </p>
          </div>
          <div className="flex flex-col gap-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-50 flex items-center justify-center">
              <Globe className="w-5 h-5 text-emerald-600" />
            </div>
            <h3 className="font-semibold text-slate-900">Deployment</h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              The trained model is served via a Gradio application hosted on Hugging Face Spaces,
              exposing a REST API that this React frontend consumes in real time.
            </p>
          </div>
        </div>

        <div className="prose prose-slate prose-sm max-w-none space-y-4 border-t border-slate-100 pt-10">
          <p className="text-slate-600 leading-relaxed">
            PneumoScan AI was developed to demonstrate how state-of-the-art deep learning
            architectures can be adapted for radiological imaging tasks with relatively modest
            computational resources. By applying transfer learning — initialising a DenseNet121
            network with ImageNet weights and fine-tuning on a domain-specific dataset — the model
            learns to distinguish pneumonia-related opacities and consolidations from healthy lung
            tissue. The DenseNet architecture's dense connectivity pattern makes it particularly
            effective at preserving fine-grained spatial features critical for X-ray interpretation.
          </p>
          <p className="text-slate-600 leading-relaxed">
            During training and evaluation, Gradient-weighted Class Activation Mapping (Grad-CAM)
            was used to visualise which regions of each X-ray the model attended to when making
            predictions. This interpretability step is essential for building trust in medical AI
            systems — it allows a human reviewer to verify that the model is focusing on clinically
            relevant anatomy rather than artefacts or image metadata. The resulting attention maps
            consistently highlighted bilateral lower-lobe regions in pneumonia-positive cases,
            aligned with clinical expectations.
          </p>
          <p className="text-slate-600 leading-relaxed">
            This tool is intended for educational use, research exploration, and portfolio
            demonstration. It is not a certified medical device. The confidence scores it produces
            reflect the model's learned probability estimates on a specific dataset and should never
            be used as the sole basis for a clinical decision. Always involve a licensed radiologist
            or physician in any real-world diagnostic workflow.
          </p>
        </div>
      </div>
    </section>
  );
}
