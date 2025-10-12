import { Button } from "@/components/ui/button"

export function AboutDeveloperSection() {
  return (
    <section aria-labelledby="about-developer-title" className="w-full border-t">
      <div className="mx-auto w-full max-w-3xl px-4 py-8 sm:py-10">
        <h2 id="about-developer-title" className="text-lg font-medium tracking-tight sm:text-xl">
          About the Developer
        </h2>

        <div className="mt-4 space-y-4 text-sm leading-relaxed text-muted-foreground">
          <p>
            I&apos;m <span className="text-foreground font-medium">Your Name</span>, a B.Tech student at VIT Vellore,
            passionate about Deep Learning and AI accessibility. I built this project to bridge culture and technology —
            bringing trustworthy, Gujarati-language answers to the “Vachanamrut.”
          </p>
          <p>
            Future plans:{" "}
            <em>
              “I aim to continue building AI systems that respect linguistic, cultural, and ethical boundaries — a
              direction I wish to deepen through the MSAII program at Carnegie Mellon.”
            </em>
          </p>

          <div className="mt-4 flex flex-wrap items-center gap-3">
            <Button asChild variant="secondary">
              <a
                href="https://github.com/yourname"
                target="_blank"
                rel="noreferrer noopener"
                aria-label="GitHub profile"
              >
                GitHub
              </a>
            </Button>
            <Button asChild variant="outline">
              <a
                href="https://www.linkedin.com/in/yourname"
                target="_blank"
                rel="noreferrer noopener"
                aria-label="LinkedIn profile"
              >
                LinkedIn
              </a>
            </Button>
          </div>
        </div>
      </div>
    </section>
  )
}
