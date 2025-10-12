// "use client"

import Image from "next/image"
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"

export function TechnicalOverviewSection() {
  return (
    <section aria-labelledby="technical-overview-title" className="w-full border-t">
      <div className="mx-auto w-full max-w-4xl px-4 py-8 sm:py-10">
        <h2 id="technical-overview-title" className="text-lg font-medium tracking-tight sm:text-xl">
          Technical Overview
        </h2>

        <div className="mt-5 grid gap-6">
          <div>
            <h3 className="text-sm font-semibold text-foreground/80 sm:text-base">System Architecture</h3>
            <div className="mt-3 overflow-hidden rounded-md border">
              <Image
                src="/system_architecture.jpeg"
                alt="High-level diagram: Gujarati RAG-based QA system architecture"
                width={800}
                height={360}
                className="h-auto w-full"
                priority
              />
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-foreground/80 sm:text-base">Tech Stack</h3>
            <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Application</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground">
                  <ul className="list-disc pl-5">
                    <li>Next.js App Router</li>
                    <li>TypeScript</li>
                    <li>Tailwind CSS + shadcn/ui</li>
                  </ul>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base">AI & RAG</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground">
                  <ul className="list-disc pl-5">
                    <li>Retrieval-Augmented Generation (RAG)</li>
                    <li>Gujarati language QA over “Vachanamrut”</li>
                    <li>Chunking, embeddings, and context retrieval</li>
                  </ul>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Serving</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground">
                  <ul className="list-disc pl-5">
                    <li>Server Actions & Route Handlers</li>
                    <li>Streaming responses for chat UX</li>
                    <li>Secure environment variables (server-only)</li>
                  </ul>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base">UX Considerations</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-muted-foreground">
                  <ul className="list-disc pl-5">
                    <li>Accessibility-first content and contrast</li>
                    <li>Mobile-first layout</li>
                    <li>Gujarati script readability</li>
                  </ul>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
