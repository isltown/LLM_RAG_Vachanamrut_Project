import { Chat } from "@/components/chat";
import { HeroSection} from "@/components/hero";

export default function Home() {
  return (
    <div className="flex flex-col size-full items-center">
      <HeroSection />
      <Chat />
    </div>
  );
}
