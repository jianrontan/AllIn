import process from 'node:process'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Fail the PRODUCTION build loudly if VITE_API_BASE is missing. Without
  // this, api.js would have silently fallen back to http://localhost:5000 --
  // a bundle that works on a dev machine and breaks for every real visitor.
  // (.env.production is committed, so this only fires if that file is
  // deleted/renamed or a CI env wipes it.)
  if (mode === 'production') {
    const env = loadEnv(mode, process.cwd(), 'VITE_')
    if (!env.VITE_API_BASE) {
      throw new Error(
        'VITE_API_BASE is not set for the production build. ' +
        'Check frontend/.env.production (committed) or the build env.')
    }
  }
  return {
    plugins: [react(), tailwindcss()],
  }
})
