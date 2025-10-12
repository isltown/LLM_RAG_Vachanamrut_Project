import { AboutDeveloperSection } from "@/components/about-developer";
import { Chat } from "@/components/chat";
import { CommunityImpactSection } from "@/components/community-impact";
import { HeroSection} from "@/components/hero";
import { MotivationSection } from "@/components/motivation";
import { TechnicalOverviewSection } from "@/components/technical-overview";

export default function Home() {
  return (
    <div className="flex flex-col size-full items-center">
      <HeroSection />
      <Chat />
    </div>
  );
}
