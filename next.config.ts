import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  eslint: {
    // Warning: This allows production builds to successfully complete even if
    // your project has ESLint errors.
    ignoreDuringBuilds: true,
  },
  transpilePackages: ['geist'],
  images: {
    remotePatterns: [
      {
        hostname: 'vercel.com',
      },
    ],
  },
};

export default nextConfig;
