import { config as dotenvConfig } from "dotenv";
import { resolve } from "path";

// Load the root .env (one level above web/) so all API routes share the same config.
// web/.env.local still takes precedence if it exists (standard Next.js behaviour).
dotenvConfig({ path: resolve(process.cwd(), "../.env"), override: false });

/** @type {import('next').NextConfig} */
const nextConfig = {};

export default nextConfig;
