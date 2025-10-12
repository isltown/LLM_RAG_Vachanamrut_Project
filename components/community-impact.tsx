import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"

export function CommunityImpactSection() {
  return (
    <section aria-labelledby="community-impact-title" className="w-full border-t">
      <div className="mx-auto w-full max-w-4xl px-4 py-8 sm:py-10">
        <h2 id="community-impact-title" className="text-lg font-medium tracking-tight sm:text-xl">
          Community Impact
        </h2>

        <p className="mt-3 text-sm text-muted-foreground">
          Helping Gujarati-speaking individuals engage with AI authentically — grounded in scripture, language, and
          trust.
        </p>

        <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">From a Parent</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              <p>
                {
                  '"I can ask questions in Gujarati and get respectful answers — it feels like the text is speaking to me."'
                }
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">From a Community Member</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              <p>{'"This made study groups easier — we confirm verses quickly and stay focused on meaning."'}</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">From a Student</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              <p>{'"Gujarati support matters. It keeps the conversation natural and the learning authentic."'}</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">From a Volunteer</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              <p>{'"It encourages more people to ask questions without worrying about language barriers."'}</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  )
}
