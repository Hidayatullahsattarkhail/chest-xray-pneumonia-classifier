import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { DisclaimerBanner } from "@/components/layout/DisclaimerBanner";
import { Hero } from "@/components/sections/Hero";
import { UploadAnalyzer } from "@/components/features/UploadAnalyzer";
import { ModelPerformance } from "@/components/sections/ModelPerformance";
import { HowItWorks } from "@/components/sections/HowItWorks";
import { SampleImages } from "@/components/sections/SampleImages";
import { About } from "@/components/sections/About";

export default function Home() {
  return (
    <div className="min-h-[100dvh] flex flex-col font-sans selection:bg-teal-100 bg-white">
      <DisclaimerBanner />
      <Navbar />

      <main className="flex-1">
        <Hero />
        <UploadAnalyzer />
        <ModelPerformance />
        <HowItWorks />
        <SampleImages />
        <About />
      </main>

      <Footer />
    </div>
  );
}
