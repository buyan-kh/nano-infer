import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The dev server rejects the HMR socket when the page is opened at
  // 127.0.0.1 and the process is bound to another host. Without that
  // socket the client never finishes hydration.
  allowedDevOrigins: ["127.0.0.1"],
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
