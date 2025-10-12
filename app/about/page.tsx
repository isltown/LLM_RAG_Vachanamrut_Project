import { AboutDeveloperSection } from '@/components/about-developer'
import { CommunityImpactSection } from '@/components/community-impact'
import { MotivationSection } from '@/components/motivation'
import { TechnicalOverviewSection } from '@/components/technical-overview'
import React from 'react'

const page = () => {
  return (
    <div className="flex flex-col size-full items-center pt-10">
    <MotivationSection />
      <TechnicalOverviewSection/>
      <CommunityImpactSection />
      <AboutDeveloperSection/>
    </div>
  )
}

export default page